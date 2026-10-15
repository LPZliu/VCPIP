from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import evaluation
import MLdataset
from model import VCPIP
from myloss import VCPIPLoss
from utils import seed_everything, setLogger


METRIC_NAMES = [
    "AP", "1-HL", "1-RL", "AUC", "1-OE", "1-Coverage",
    "macro_AUC", "macro_F1", "micro_F1",
]


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but it is unavailable; use --device cpu.")
    return device


def make_logger(output_dir):
    return setLogger(Path(output_dir) / "run.log")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def move_batch(batch, device):
    views, labels, view_mask, label_mask = batch
    return (
        [view.to(device=device, dtype=torch.float32) for view in views],
        labels.to(device=device, dtype=torch.float32),
        view_mask.to(device=device, dtype=torch.float32),
        label_mask.to(device=device, dtype=torch.float32),
    )


@contextmanager
def singleton_batchnorm_guard(model, batch_size):
    layers = [
        layer for layer in model.modules()
        if batch_size == 1 and isinstance(layer, nn.modules.batchnorm._BatchNorm)
        and layer.training
    ]
    for layer in layers:
        layer.eval()
    try:
        yield
    finally:
        for layer in layers:
            layer.train()


def train_one_epoch(loader, model, criterion, optimizer, device, eta=0.2, max_batches=0):
    model.train()
    totals = {}
    samples = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        views, labels, view_mask, label_mask = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with singleton_batchnorm_guard(model, labels.shape[0]):
            outputs = model(views, view_mask, eta=eta)
            losses = criterion(outputs, views, labels, view_mask, label_mask)
        if not all(torch.isfinite(value).all().item() for value in losses.values()):
            raise FloatingPointError(f"Nonfinite training loss in batch {batch_index}.")
        losses["total"].backward()
        optimizer.step()
        count = labels.shape[0]
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
        samples += count
        batches += 1
    if samples == 0:
        raise ValueError("The training loader contains no samples.")
    return {
        "losses": {name: total / samples for name, total in totals.items()},
        "samples": samples,
        "batches": batches,
        "partial_training_epoch": bool(max_batches and max_batches < len(loader)),
    }


def compute_view_confidence(view_probabilities, labels, view_mask, label_mask):
    statistics = []
    for view_index, probability in enumerate(view_probabilities):
        valid = label_mask * view_mask[:, view_index:view_index + 1]
        positives = valid * labels
        negatives = valid * (1.0 - labels)
        statistics.append(torch.stack([
            (probability * positives).sum(), positives.sum(),
            ((1.0 - probability) * negatives).sum(), negatives.sum(),
        ]))
    return torch.stack(statistics).detach().to(dtype=torch.float64, device="cpu")


@torch.no_grad()
def evaluate(loader, model, device, max_batches=0):
    model.eval()
    predictions, labels_all = [], []
    confidence_stats = None
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        views, labels, view_mask, label_mask = move_batch(batch, device)
        outputs = model(views, view_mask, eta=0.0)
        predictions.append(outputs["predictions"].detach().cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        statistics = compute_view_confidence(
            outputs["view_probabilities"], labels, view_mask, label_mask,
        )
        confidence_stats = statistics if confidence_stats is None else confidence_stats + statistics
        batches += 1
    if not predictions:
        raise ValueError("The evaluation loader contains no samples.")
    scores = np.asarray(evaluation.do_metric(
        np.concatenate(predictions), np.concatenate(labels_all),
    ), dtype=float)
    confidence = []
    supported = []
    for positive_sum, positive_count, negative_sum, negative_count in confidence_stats.tolist():
        valid = positive_count > 0 and negative_count > 0
        supported.append(valid)
        confidence.append(
            positive_sum / positive_count + negative_sum / negative_count if valid else None
        )
    return {
        "metrics": dict(zip(METRIC_NAMES, scores.tolist())),
        "confidence": confidence,
        "confidence_supported": supported,
        "confidence_statistics": confidence_stats.tolist(),
        "samples": sum(len(batch) for batch in predictions),
        "batches": batches,
        "partial_evaluation": bool(max_batches and max_batches < len(loader)),
    }


def evaluate_and_add_layers(model, validation, optimizer, threshold=0.99, max_view_layers=3):
    supported = [
        (index, score) for index, score in enumerate(validation["confidence"])
        if score is not None and np.isfinite(score)
    ]
    if len(supported) < 2:
        return {"expanded": False, "reason": "fewer_than_two_supported_views"}
    view_index, minimum = min(supported, key=lambda item: item[1])
    maximum = max(score for _, score in supported)
    if maximum <= 0:
        return {"expanded": False, "reason": "zero_confidence"}
    ratio = minimum / maximum
    event = {"expanded": False, "view": view_index, "ratio": ratio}
    if ratio >= threshold:
        return {**event, "reason": "confidence_ratio_above_threshold"}
    if model.get_view_layer_count(view_index) >= max_view_layers:
        return {**event, "reason": "weakest_view_at_stage_limit"}
    existing_ids = {id(parameter) for parameter in model.parameters()}
    model.add_classifier_layer(view_index)
    new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in existing_ids]
    if not new_parameters:
        raise RuntimeError("A new prediction stage did not register any parameters.")
    optimizer.add_param_group({"params": new_parameters})
    return {
        **event, "expanded": True,
        "total_stages": model.get_view_layer_count(view_index),
        "new_parameters": sum(parameter.numel() for parameter in new_parameters),
    }


def save_checkpoint(path, model, model_config, *, train_config=None, epoch=0,
                    validation=None, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_config": dict(model_config),
        "boost_counts": list(model.boost_counts),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "train_config": _json_safe(train_config or {}),
        "epoch": int(epoch),
        "validation": _json_safe(validation),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use a checkpoint created by main.py.")
    model = VCPIP(**checkpoint["model_config"], boost_counts=checkpoint["boost_counts"])
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def dataset_paths(args):
    directory = Path(args.data_root) / args.dataset
    data_path = directory / f"{args.dataset}_six_view.mat"
    fold_path = directory / (
        f"{args.dataset}_six_view_MaskRatios_{args.mask_view_ratio:g}"
        f"_LabelMaskRatio_{args.mask_label_ratio:g}"
        f"_TraindataRatio_{args.training_sample_ratio:g}.mat"
    )
    for path in (data_path, fold_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}. See data/README.md.")
    return data_path, fold_path


def build_loader(args, paths, fold_index, mode):
    return MLdataset.getIncDataloader(
        str(paths[0]), str(paths[1]), training_ratio=args.training_sample_ratio,
        val_ratio=args.validation_ratio, fold_idx=fold_index, mode=mode,
        batch_size=args.batch_size, num_workers=args.workers, shuffle=mode == "train",
        normalization=args.normalization,
    )


def selection_score(validation):
    metrics = validation["metrics"]
    values = [metrics["AP"], metrics["1-RL"], metrics["AUC"]]
    if not all(value is not None and np.isfinite(value) for value in values):
        raise ValueError("Validation AP, 1-RL, or AUC is undefined; check the split and labels.")
    return 0.25 * values[0] + 0.25 * values[1] + 0.5 * values[2]


def train(args, output_dir, logger, device):
    paths = dataset_paths(args)
    fold_results = []
    for fold_index in range(args.fold_index, args.fold_index + args.folds_num):
        seed_everything(args.seed + fold_index)
        train_loader, train_dataset = build_loader(args, paths, fold_index, "train")
        validation_loader, _ = build_loader(args, paths, fold_index, "val")
        test_loader, _ = build_loader(args, paths, fold_index, "test")
        model_config = {
            "input_dims": list(map(int, train_dataset.d_list)),
            "n_labels": int(train_dataset.classes_num), "num_experts": args.num_experts,
        }
        model = VCPIP(**model_config).to(device)
        criterion = VCPIPLoss(alpha=args.alpha, beta=args.beta).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        checkpoint_path = output_dir / "checkpoints" / f"fold_{fold_index}_best.pt"
        history = []
        best_score = -float("inf")
        logger.info("Fold %d: %d training samples, %d views, %d labels", fold_index,
                    len(train_dataset), len(model_config["input_dims"]), model_config["n_labels"])
        for epoch in range(args.epochs):
            training = train_one_epoch(
                train_loader, model, criterion, optimizer, device,
                eta=args.eta, max_batches=args.max_train_batches,
            )
            validation = evaluate(validation_loader, model, device, args.max_eval_batches)
            score = selection_score(validation)
            if score >= best_score:
                best_score = score
                save_checkpoint(
                    checkpoint_path, model, model_config,
                    train_config={**vars(args), "fold_index": fold_index},
                    epoch=epoch, validation=validation,
                )
            growth = (
                evaluate_and_add_layers(model, validation, optimizer, args.threshold, args.max_view_layers)
                if epoch + 1 < args.epochs else {"expanded": False, "reason": "last_epoch"}
            )
            history.append({"epoch": epoch, "training": training, "validation": validation,
                            "selection_score": score, "growth": growth})
            write_json(output_dir / f"fold_{fold_index}_history.json", history)
            logger.info("Epoch %d | loss %.5f | val AP %.5f | 1-RL %.5f | AUC %.5f | stages %s",
                        epoch, training["losses"]["total"], validation["metrics"]["AP"],
                        validation["metrics"]["1-RL"], validation["metrics"]["AUC"],
                        [model.get_view_layer_count(index) for index in range(len(model_config["input_dims"]))])
            if growth["expanded"]:
                logger.info("Expanded view %d to %d stages for the next epoch (ratio %.5f).",
                            growth["view"], growth["total_stages"], growth["ratio"])
        best_model, checkpoint = load_checkpoint(checkpoint_path, device)
        result = {
            "fold_index": fold_index, "best_epoch": checkpoint["epoch"],
            "checkpoint": str(checkpoint_path),
            "test": evaluate(test_loader, best_model, device, args.max_eval_batches),
            "training_was_bounded": bool(args.max_train_batches and args.max_train_batches < len(train_loader)),
        }
        fold_results.append(result)
        write_json(output_dir / "results.json", {"folds": fold_results})
        logger.info("Fold %d best epoch %d | test %s", fold_index, checkpoint["epoch"], result["test"]["metrics"])
    aggregate = {}
    for name in METRIC_NAMES:
        values = [result["test"]["metrics"][name] for result in fold_results]
        finite = [value for value in values if value is not None and np.isfinite(value)]
        aggregate[name] = {
            "mean": float(np.mean(finite)) if finite else None,
            "std": float(np.std(finite)) if finite else None,
            "valid_folds": len(finite),
        }
    summary = {"folds": fold_results, "aggregate": aggregate,
               "bounded_run": any(result["training_was_bounded"] or result["test"]["partial_evaluation"]
                                  for result in fold_results)}
    write_json(output_dir / "results.json", summary)
    return summary


def smoke_test(args, output_dir, device):
    seed_everything(args.seed)
    model_config = {"input_dims": [8, 10, 6], "n_labels": 4, "num_experts": 2}
    model = VCPIP(**model_config).to(device)
    criterion = VCPIPLoss(alpha=args.alpha, beta=args.beta).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    views = [torch.randn(6, width, device=device) for width in model_config["input_dims"]]
    view_mask = torch.ones(6, 3, device=device)
    view_mask[0, 1] = view_mask[1, 2] = 0
    views = [view * view_mask[:, index:index + 1] for index, view in enumerate(views)]
    originals = [view.clone() for view in views]
    labels = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]] * 3, device=device, dtype=torch.float32)
    label_mask = torch.ones_like(labels)
    label_mask[0, 2] = 0
    model.train()
    outputs = model(views, view_mask, eta=args.eta)
    losses = criterion(outputs, views, labels, view_mask, label_mask)
    if not all(torch.isfinite(loss).all().item() for loss in losses.values()):
        raise AssertionError("The synthetic objective is not finite.")
    losses["total"].backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
               for parameter in model.parameters()):
        raise AssertionError("A synthetic gradient is not finite.")
    optimizer.step()
    if not all(torch.equal(before, after) for before, after in zip(originals, views)):
        raise AssertionError("SCFS changed the original reconstruction targets in place.")
    existing_ids = {id(parameter) for parameter in model.parameters()}
    growth = evaluate_and_add_layers(
        model, {"confidence": [1.8, 0.6, 1.7]}, optimizer,
        threshold=0.99, max_view_layers=3,
    )
    if not growth["expanded"]:
        raise AssertionError("The synthetic weakest view was not expanded.")
    optimizer.zero_grad(set_to_none=True)
    expanded_loss = criterion(model(views, view_mask, eta=0.0), views, labels, view_mask, label_mask)
    expanded_loss["total"].backward()
    new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in existing_ids]
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
               for parameter in new_parameters):
        raise AssertionError("New stage parameters do not receive finite gradients.")
    checkpoint_path = output_dir / "smoke_checkpoint.pt"
    save_checkpoint(checkpoint_path, model, model_config, train_config={"synthetic_only": True})
    restored, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model(views, view_mask, eta=0.0)["predictions"]
        actual = restored(views, view_mask, eta=0.0)["predictions"]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    result = {
        "passed": True, "synthetic_only": True, "device": str(device),
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "growth": growth, "boost_counts": checkpoint["boost_counts"],
        "checkpoint_prediction_max_abs_difference": float((expected - actual).abs().max()),
        "checks": ["finite_objective", "finite_gradients", "unmodified_inputs",
                   "dynamic_stage_registration", "new_stage_gradients", "checkpoint_equivalence"],
    }
    write_json(output_dir / "smoke_results.json", result)
    return result


def parse_args(argv=None):
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate VCPIP.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="corel5k")
    parser.add_argument("--data-root", type=Path, default=base / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--folds-num", type=int, default=1)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1.0, help="Orthogonality coefficient")
    parser.add_argument("--beta", type=float, default=0.1, help="Reconstruction coefficient")
    parser.add_argument("--threshold", type=float, default=0.99, help="VCA expansion threshold tau")
    parser.add_argument("--eta", type=float, default=0.2, help="SCFS fragment proportion")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--max-view-layers", type=int, default=3, help="Maximum total stages per view")
    parser.add_argument("--mask-view-ratio", type=float, default=0.5)
    parser.add_argument("--mask-label-ratio", type=float, default=0.5)
    parser.add_argument("--training-sample-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--normalization", choices=("train", "legacy-all"), default="train",
                        help="Fit normalization on observed training views or all samples")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Maximum training batches; 0 means all batches")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means all batches; capped metrics are partial")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Evaluate this saved model only; no training")
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic gradient/growth/checkpoint check only")
    args = parser.parse_args(argv)
    for name in ("epochs", "folds_num", "batch_size", "num_experts", "max_view_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("workers", "fold_index", "max_train_batches", "max_eval_batches", "seed"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if not 0 <= args.eta <= 1 or not 0 < args.threshold <= 1:
        parser.error("--eta must be in [0, 1] and --threshold must be in (0, 1]")
    if args.alpha < 0 or args.beta < 0 or args.lr <= 0 or args.momentum < 0:
        parser.error("Loss weights and momentum must be nonnegative; learning rate must be positive")
    if not (0 < args.training_sample_ratio < 1 and 0 < args.validation_ratio < 1
            and args.training_sample_ratio + args.validation_ratio < 1):
        parser.error("Training and validation ratios must be positive and sum to less than 1")
    if not 0 <= args.mask_view_ratio <= 1 or not 0 <= args.mask_label_ratio <= 1:
        parser.error("Mask ratios must be in [0, 1]")
    if args.smoke_test and args.checkpoint:
        parser.error("Use either --smoke-test or --checkpoint")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    model, checkpoint = None, None
    if args.checkpoint:
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        saved_config = checkpoint.get("train_config", {})
        data_fields = (
            "dataset", "mask_view_ratio", "mask_label_ratio", "training_sample_ratio",
            "validation_ratio", "normalization", "fold_index",
        )
        missing_fields = [field for field in data_fields if field not in saved_config]
        if missing_fields:
            raise ValueError(
                "This checkpoint lacks the dataset metadata required for evaluation: "
                + ", ".join(missing_fields)
            )
        for field in data_fields:
            setattr(args, field, saved_config[field])
        args.folds_num = 1
    output_dir = args.output_dir or Path(__file__).resolve().parent / "records" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"Output directory must be new or empty to preserve previous runs: {output_dir}. "
            "Choose a new --output-dir or omit it for a unique timestamped run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    logger = make_logger(output_dir)
    write_json(output_dir / "config.json", {**vars(args), "resolved_device": str(device)})
    logger.info("VCPIP | device %s | outputs %s", device, output_dir)
    if args.smoke_test:
        result = smoke_test(args, output_dir, device)
        logger.info("Smoke test passed.")
    elif args.checkpoint:
        fold_index = args.fold_index
        logger.info("Dataset, split, masks, normalization, and fold restored from the checkpoint.")
        loader, dataset = build_loader(args, dataset_paths(args), fold_index, "test")
        if list(dataset.d_list) != checkpoint["model_config"]["input_dims"] or dataset.classes_num != checkpoint["model_config"]["n_labels"]:
            raise ValueError("The selected dataset dimensions do not match this checkpoint.")
        result = {"checkpoint": str(args.checkpoint), "fold_index": fold_index,
                  "test": evaluate(loader, model, device, args.max_eval_batches)}
        write_json(output_dir / "results.json", result)
        logger.info("Evaluation only | %s", result["test"]["metrics"])
    else:
        result = train(args, output_dir, logger, device)
    logger.info("Saved results to %s", output_dir)
    return result


if __name__ == "__main__":
    main()

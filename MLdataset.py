import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


def load_incomplete_fold(mat_path, fold_mat_path, fold_idx=0,
                         training_ratio=0.7, normalization="train"):
    for path in (mat_path, fold_mat_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dataset file missing: {path}. See data/README.md.")
    data, folds = loadmat(mat_path), loadmat(fold_mat_path)
    views = [np.asarray(v, dtype=np.float32) for v in data["X"].ravel()]
    labels = np.asarray(data["label"], dtype=np.float32)
    indices = folds["folds_sample_index"].ravel()
    if not 0 <= fold_idx < len(indices):
        raise ValueError(f"fold_idx must be in [0, {len(indices) - 1}]")
    order = np.asarray(indices[fold_idx], dtype=np.int64).ravel() - 1
    n = len(order)
    if sorted(order.tolist()) != list(range(n)):
        raise ValueError("folds_sample_index must be a 1-based permutation of all samples")
    if labels.shape[0] != n and labels.shape[1] == n:
        labels = labels.T
    views = [v.T if v.shape[0] != n and v.shape[1] == n else v for v in views]
    if labels.shape[0] != n or any(v.shape[0] != n for v in views):
        raise ValueError("Feature, label and fold sample dimensions do not agree")
    if np.min(labels) == -1:
        labels = (labels + 1) / 2
    view_mask = np.asarray(folds["folds_data"].ravel()[fold_idx], dtype=np.float32)
    label_mask = np.asarray(folds["folds_label"].ravel()[fold_idx], dtype=np.float32)
    if view_mask.shape != (n, len(views)) or label_mask.shape != labels.shape:
        raise ValueError("Missing-view or missing-label mask has an incompatible shape")
    if any(not np.isin(x, (0, 1)).all() for x in (labels, view_mask, label_mask)):
        raise ValueError("Labels and availability masks must be binary")
    if not all(np.isfinite(x).all() for x in views):
        raise ValueError("Input features contain NaN or infinity")
    labels, view_mask, label_mask = labels[order], view_mask[order], label_mask[order]
    n_train = math.ceil(n * training_ratio)
    if not 0 < n_train < n:
        raise ValueError("training_ratio must leave both training and held-out samples")
    normalized = []
    for v, features in enumerate(views):
        features = features[order]
        if normalization == "train":
            observed = view_mask[:n_train, v].astype(bool)
            if not observed.any():
                raise ValueError(f"View {v} has no observed training sample to fit normalization")
            fit_data = features[:n_train][observed]
        elif normalization == "legacy-all":
            fit_data = features
        else:
            raise ValueError("normalization must be 'train' or 'legacy-all'")
        transformed = StandardScaler().fit(fit_data).transform(features)
        normalized.append(np.asarray(transformed * view_mask[:, v:v + 1], dtype=np.float32))
    return normalized, labels, view_mask, label_mask, order


class IncDataset(Dataset):
    def __init__(self, mat_path, fold_mat_path, training_ratio=0.7, val_ratio=0.15,
                 fold_idx=0, mode="train", normalization="train", **unused):
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be train, val or test")
        if training_ratio <= 0 or val_ratio <= 0 or training_ratio + val_ratio >= 1:
            raise ValueError("Positive train/validation ratios must sum to less than 1")
        views, labels, view_mask, label_mask, order = load_incomplete_fold(
            mat_path, fold_mat_path, fold_idx, training_ratio, normalization)
        n = len(labels)
        self.train_sample_num = math.ceil(n * training_ratio)
        self.val_sample_num = math.ceil(n * val_ratio)
        self.test_sample_num = n - self.train_sample_num - self.val_sample_num
        if self.test_sample_num < 1:
            raise ValueError("Split rounding leaves no test sample")
        edges = {"train": (0, self.train_sample_num),
                 "val": (self.train_sample_num, self.train_sample_num + self.val_sample_num),
                 "test": (self.train_sample_num + self.val_sample_num, n)}
        start, stop = edges[mode]
        self.cur_mv_data = [v[start:stop] for v in views]
        self.cur_inc_V_ind = view_mask[start:stop]
        self.cur_inc_L_ind = label_mask[start:stop] if mode == "train" else np.ones_like(labels[start:stop])
        self.cur_labels = labels[start:stop] * self.cur_inc_L_ind
        self.sample_indices = order[start:stop]
        self.classes_num, self.d_list = labels.shape[1], [v.shape[1] for v in views]
        self.mode = mode

    def __len__(self):
        return len(self.cur_labels)

    def __getitem__(self, index):
        return ([torch.from_numpy(v[index]) for v in self.cur_mv_data],
                torch.from_numpy(self.cur_labels[index]),
                torch.from_numpy(self.cur_inc_V_ind[index]),
                torch.from_numpy(self.cur_inc_L_ind[index]))


def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=0.7,
                     val_ratio=0.15, fold_idx=0, mode="train", batch_size=128,
                     num_workers=0, shuffle=False, normalization="train"):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio, val_ratio,
                         fold_idx, mode, normalization)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=False)
    return loader, dataset

# VCPIP (NeurIPS 2026)

Code for the **NeurIPS 2026 (NIPS 2026)** paper **View Confidence Perception-Driven Incremental Prediction for Incomplete Multi-view Multi-label Learning**.

VCPIP combines validation-guided incremental classifiers, mixture-of-experts decoupling and fusion, and stochastic cross-sample fragment swapping for incomplete multi-view multi-label classification.

This release aligns the implementation with Eqs. (1)–(12) of the manuscript. It is a new algorithm version: the previous experiment scripts and their results are not evidence for this version. No benchmark retraining was performed for this release. Two details that the manuscript does not fully specify—the probability parameterization of incremental classifiers and the compatible feature dimension in Eq. (10)—are stated in [implementation notes](docs/IMPLEMENTATION_NOTES.md).

## Repository structure

The compact research-code layout follows [LMVCAT](https://github.com/justsmart/LMVCAT).

```text
VCPIP/
├── main.py                  # Training, validation, dynamic expansion, and evaluation
├── model.py                 # SCFS, MOEDF, incremental classifiers, and final prediction
├── myloss.py                # Paper losses: Eqs. (4)–(7), (11), and (12)
├── MLdataset.py             # MATLAB data, split/mask loading, and normalization
├── evaluation.py            # Multi-label evaluation metrics
├── utils.py                 # Reproducibility and supporting utilities
├── requirements.txt
├── data/                    # Local data; dataset files are not committed
├── records/                 # Local run outputs; generated files are not committed
├── docs/
│   ├── METHOD_CODE_MAP.md    # Contribution-by-contribution and equation-by-equation map
│   ├── IMPLEMENTATION_NOTES.md
│   ├── PROVENANCE.md
│   └── VALIDATION.md         # Checks completed for this release
└── tests/
    ├── test_method.py       # Formula and implementation checks
    └── test_data.py         # Data, normalization, and metric checks
```

## Requirements

Python 3.9 or later and PyTorch 2.6 or later are required. Install the appropriate PyTorch build for your CPU or CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Dependency checks for this release used Python 3.9.23, PyTorch 2.8.0+cu129, NumPy 2.0.2, SciPy 1.13.1, and scikit-learn 1.6.1. This records the checked environment, not a benchmark training environment.

## Data preparation

Place the feature/label file and its pre-generated incomplete-data split file in the corresponding dataset directory. For the default Corel5k setting:

```text
data/corel5k/
├── corel5k_six_view.mat
└── corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat
```

The source bundle's README points to this [dataset location](https://drive.google.com/drive/folders/1ey17GpSJEYpYchY6Du_AOj5Yzi2Ml7JU?usp=drive_link). Dataset files, checkpoints, the supplied manuscript PDF, and historical experiment outputs are not distributed in this repository.

The feature file contains MATLAB variables `X` (a cell array of view matrices) and `label`; the split file contains `folds_sample_index`, `folds_data`, and `folds_label`. See [data preparation](data/README.md) for the expected format.

Features are standardized using observed training samples only. The same fitted transformation is applied to validation and test samples. The existing fold permutation and missingness masks are loaded from the split file; subset boundaries use the selected train/validation ratios. Training uses observed labels; validation and test use the complete held-out labels, following the supplied data protocol.

## Run

After preparing the data, the default entry point is:

```bash
python main.py
```

This command starts training with the local Corel5k files. An explicit equivalent with a chosen output directory is:

```bash
python main.py --dataset corel5k --data-root data --device auto --output-dir records/corel5k
```

To check synthetic forward/backward behavior, dynamic expansion, and checkpoint loading without loading or training a benchmark dataset:

```bash
python main.py --smoke-test --device cpu --output-dir records/smoke
```

Evaluate a checkpoint created by this version:

```bash
python main.py --checkpoint records/corel5k/checkpoints/fold_0_best.pt --data-root data --device auto --output-dir records/evaluation
```

Evaluation restores the dataset name, missingness ratios, split ratios, normalization, fold index, and dynamic classifier counts from the checkpoint. Use `--data-root` to point to the same dataset files after moving them; runtime options such as `--device`, `--batch-size`, and `--workers` remain configurable. A synthetic smoke-test checkpoint cannot be used as a benchmark evaluation checkpoint.

Inspect all data, device, optimization, and evaluation options with:

```bash
python main.py --help
```

Run the implementation checks without launching benchmark training:

```bash
python -m unittest discover -s tests -v
```

Runs save configuration, logs, per-epoch history, results, and the selected checkpoint under `records/` or `--output-dir`. Explicit output directories must be new or empty; omit `--output-dir` to create a fresh timestamped directory. The CLI defaults to 200 epochs, one fold, and at most three total classifier stages per view; these are implementation settings rather than manuscript-specified constants. Nonzero `--max-train-batches` or `--max-eval-batches` produces a partial diagnostic run, not a benchmark result.

See [validation](docs/VALIDATION.md) for the checks completed for this release and their limits.

## Method-to-code correspondence

| Contribution | Paper | Implementation |
| --- | --- | --- |
| View-confidence-aware incremental prediction | Sec. 3.1; Eqs. (1)–(5) | Validation confidence and branch expansion in `main.py`; incremental classifiers in `model.py`; residual and all-stage supervision in `myloss.py` |
| Mixture-of-experts decouple-fusion (MOEDF) | Sec. 3.2; Eqs. (6)–(7) | `MoEDecouple` and `MoEFuser` in `model.py`; orthogonality and reconstruction in `myloss.py` |
| Stochastic cross-sample fragment swapping (SCFS) | Sec. 3.3; Eq. (8) | `stochastic_cross_sample_fragment_swap` in `model.py`; augmentation during training only |
| Confidence-weighted prediction and total objective | Sec. 3.4; Eqs. (9)–(12) | `VCPIP` in `model.py`; the four-term objective in `myloss.py` |

The total objective is exactly the four-term structure specified by Eq. (12):

$$\mathcal L = \mathcal L_{mc}+\mathcal L_{mv}+\alpha\mathcal L_{orth}+\beta\mathcal L_{re}.$$

Default manuscript settings are SGD, learning rate `0.1`, momentum `0.9`, batch size `128`, `alpha=1`, `beta=0.1`, confidence threshold `tau=0.99`, swapping proportion `eta=0.2`, and four experts. See the [bilingual method/code map](docs/METHOD_CODE_MAP.md) for exact functions and the [implementation notes](docs/IMPLEMENTATION_NOTES.md) for mathematical conventions and validation boundaries.

## Citation and attribution

If you find this code useful, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{liu2026vcpip,
  title={View Confidence Perception-Driven Incremental Prediction for Incomplete Multi-view Multi-label Learning},
  author={Liu, Pingzhu and Xu, Zunnan and He, Chunming and Fang, Zhirui and Yu, Zitong and Li, Xiu and Liu, Chengliang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

The local development starting point was the supplied RANK-derived code bundle. See [provenance and attribution](docs/PROVENANCE.md) for the original paper, retained source attribution, and licensing status.

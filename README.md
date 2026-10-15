# VCPIP (NeurIPS 2026)

**View Confidence Perception-Driven Incremental Prediction for Incomplete Multi-view Multi-label Learning**

VCPIP combines validation-guided incremental classifiers, mixture-of-experts decoupling and fusion, and stochastic cross-sample fragment swapping for incomplete multi-view multi-label classification.

## Repository structure

```text
VCPIP/
├── main.py
├── model.py
├── myloss.py
├── MLdataset.py
├── evaluation.py
├── utils.py
├── requirements.txt
├── data/
├── records/
├── docs/
│   └── PROVENANCE.md
└── tests/
    ├── test_method.py
    └── test_data.py
```

## Requirements

Python 3.9 or later and PyTorch 2.6 or later are required. Install the appropriate PyTorch build for your CPU or CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Download the datasets from [the dataset folder](https://drive.google.com/drive/folders/1ey17GpSJEYpYchY6Du_AOj5Yzi2Ml7JU?usp=drive_link). Place the feature/label file and its incomplete-data split file in the corresponding dataset directory. For the default Corel5k setting:

```text
data/corel5k/
├── corel5k_six_view.mat
└── corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat
```

The feature file contains MATLAB variables `X` and `label`; the split file contains `folds_sample_index`, `folds_data`, and `folds_label`. See [data preparation](data/README.md) for the expected format.

Features are standardized using observed training samples. The same transformation is applied to validation and test samples. Training uses observed labels; validation and test use complete held-out labels.

## Training

```bash
python main.py
```

To choose the dataset, device, and output directory:

```bash
python main.py --dataset corel5k --data-root data --device auto --output-dir records/corel5k
```

The default settings use SGD with learning rate `0.1`, momentum `0.9`, batch size `128`, `alpha=1`, `beta=0.1`, `tau=0.99`, `eta=0.2`, and four experts. Training runs for 200 epochs on one fold, with up to three classifier stages per view.

Runs save configuration, logs, per-epoch history, results, and checkpoints under `records/` or `--output-dir`. An explicit output directory must be new or empty; omit `--output-dir` to create a timestamped directory.

## Evaluation

```bash
python main.py --checkpoint records/corel5k/checkpoints/fold_0_best.pt --data-root data --device auto --output-dir records/evaluation
```

Evaluation restores the dataset, split, normalization, fold, and dynamic classifier configuration from the checkpoint. Use `--data-root` when relocating dataset files. Device, batch size, and worker count remain configurable.

## Options and tests

```bash
python main.py --help
python main.py --smoke-test --device cpu --output-dir records/smoke
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@inproceedings{liu2026vcpip,
  title={View Confidence Perception-Driven Incremental Prediction for Incomplete Multi-view Multi-label Learning},
  author={Liu, Pingzhu and Xu, Zunnan and He, Chunming and Fang, Zhirui and Yu, Zitong and Li, Xiu and Liu, Chengliang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgments

We acknowledge [RANK](https://arxiv.org/pdf/2303.17117) for the data and evaluation foundation, and [LMVCAT](https://github.com/justsmart/LMVCAT) for the repository layout reference. See [attribution](docs/PROVENANCE.md).

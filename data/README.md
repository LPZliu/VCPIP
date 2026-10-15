# Data preparation

Download the datasets from [the dataset folder](https://drive.google.com/drive/folders/1ey17GpSJEYpYchY6Du_AOj5Yzi2Ml7JU?usp=drive_link).

For Corel5k, arrange the files as follows:

```text
data/
  corel5k/
    corel5k_six_view.mat
    corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat
```

Other supported dataset names are `pascal07`, `espgame`, `iaprtc12`, and `mirflickr`.

The feature MAT file contains `X`, a MATLAB cell array of sample-by-feature view matrices, and `label`, a sample-by-category binary matrix. The fold MAT file contains `folds_data`, `folds_label`, and `folds_sample_index`: MATLAB cell arrays of view masks, label masks, and 1-based sample permutations. A mask value of 1 means observed. Labels encoded as -1/+1 are converted to 0/1.

The default split uses 70% training, 15% validation, and the remaining samples for testing. Fold permutations and missingness masks are loaded from the MAT files. Training uses observed labels; validation and test use complete held-out labels. View missingness is retained in all splits.

Normalization is fitted on observed training samples of each view. The optional `--normalization legacy-all` setting uses all samples to fit normalization.

The loader reads standard MATLAB files through SciPy. Convert MATLAB v7.3 HDF5 files to this format before loading them.

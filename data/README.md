# Data preparation

Dataset files are kept locally and are not included in this repository.
The data link from the original RANK README is
[the authors' dataset folder](https://drive.google.com/drive/folders/1ey17GpSJEYpYchY6Du_AOj5Yzi2Ml7JU?usp=drive_link).
Download/access availability and dataset redistribution permissions are not verified here.

For the default Corel5k experiment, use:

```text
data/
  corel5k/
    corel5k_six_view.mat
    corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat
```

Other supported names are `pascal07`, `espgame`, `iaprtc12` and `mirflickr`.
The feature MAT contains `X` (MATLAB cell array of sample-by-feature view
matrices) and `label` (sample-by-category binary matrix). The fold MAT contains
`folds_data`, `folds_label`, and `folds_sample_index`: MATLAB cell arrays of
view masks, label masks, and 1-based sample permutations. A value of 1 means
observed. Labels encoded as -1/+1 are converted to 0/1.

The default split uses 70% training, 15% validation and the remaining samples
for testing. Fold permutations are supplied by the MAT files. Validation and
test use full labels, while training uses only known labels. View missingness
is retained in all splits. Normalization is fitted on observed training samples
of each view; `--normalization legacy-all` restores the supplied bundle's
all-sample normalization protocol for explicit comparisons.

The loader reads standard MATLAB files through SciPy; MATLAB v7.3 HDF5 files
must first be converted to this format. Data download is not automated.

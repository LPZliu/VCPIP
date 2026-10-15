# Implementation notes

## Scope and evidence

The algorithmic specification is the VCPIP manuscript, Sections 3.1–3.4 and Appendices C, D, and G. The local source bundle was a development starting point, but its active objectives and prediction paths differed from the manuscript. This revision makes those changes explicit.

No benchmark retraining was performed for this release. The repository's checks cover executable behavior; they do not verify the manuscript's performance tables. Historical logs and checkpoints from the source bundle are not results for this revision.

## Paper defaults

| Setting | Value | Source |
| --- | --- | --- |
| Optimizer | SGD | Algorithm 1; Appendix D |
| Initial learning rate | 0.1 | Appendix D |
| Momentum | 0.9 | Appendix D |
| Batch size | 128 | Appendix D |
| Orthogonality coefficient `alpha` | 1.0 | Eq. (12); Appendix G |
| Reconstruction coefficient `beta` | 0.1 | Eq. (12); Appendix G |
| Expansion threshold `tau` | 0.99 | Eq. (2); Appendix G |
| Swapping proportion `eta` | 0.2 | Section 3.3; Appendix G |
| Experts | 4 | Appendix G |
| Standard missing-view / missing-label rates | 0.5 / 0.5 | Appendix C |
| Standard train / validation / test proportions | 0.70 / 0.15 / 0.15 | Appendix C |

The paper describes 100 epochs for the timing comparison in Appendix B. It does not establish that one training schedule reproduces every reported table. Supplied data split files remain the source of actual sample membership and missingness.

The CLI defaults to 200 epochs, one fold starting at index 0, seed 1, and at most three total classifier stages per view. These are exposed implementation settings, not constants established by the manuscript. The validation checkpoint criterion is the inherited `0.25 * AP + 0.25 * (1 - RL) + 0.5 * AUC`; the paper does not specify this selection rule.

## Incremental probabilities and residual targets

Eqs. (4)–(5) describe individual classifier probabilities, cumulative sums, and BCE supervision, but do not specify how cumulative probabilities are constrained. Directly summing independent probabilities can produce values greater than one and negative positive-label residuals.

This implementation uses:

```text
previous = 0
increment_t = (1 - previous) * sigmoid(head_t(features))
cumulative_t = previous + increment_t
```

Thus every cumulative prediction lies in `[0, 1]`; Eq. (4) becomes `Y * (1 - previous)` and remains a valid BCE target without silently clipping a negative residual. The target is detached from gradient computation, while classifier predictions retain their gradients; treating the residual as a fixed supervision target is an implementation convention. Each view starts with one base classifier and zero auxiliary classifiers. For a boosted view, Eq. (5) supervises all cumulative states and the current residual increment; for an unboosted view, it uses only the base BCE. Classifiers read encoded features `Z`; MOEDF's refined `Z'` supplies the structural term in Eq. (10).

**Bounded increments are an implementation convention that the manuscript does not define.** They make the published probability equations executable, but are not evidence of the historical experiments' exact parameterization. Numerical protection for BCE logs is separate from changing the mathematical residual target.

## Final fusion and representation dimensions

Eq. (10) is implemented in this order:

```text
semantic = sum_v(view_weight_v * cumulative_probability_v)
refined = sum_v(view_weight_v * refined_feature_v)
fused = relu(semantic * softmax(refined, dim=feature))
prediction = sigmoid(final_linear(fused))
```

The manuscript defines `c` label dimensions and `d_e` latent dimensions but multiplies the two spaces without defining a projection. Here `d_e = c`, so the multiplication is well-defined without introducing an extra projection. **The equality is an implementation choice, not a dimension specified in the paper.** Changing it requires an explicit dimension-matching rule and constitutes another implementation variant.

View weights are masked and renormalized over available views. For an all-missing sample, every view weight is zero and the final prediction depends only on the final layer's bias. Such inputs are handled without NaNs, although Appendix C's standard data construction preserves at least one view per sample.

## Losses and masking

The total loss contains only the four terms from Eq. (12):

```text
classification + multiview + alpha * orthogonality + beta * reconstruction
```

- Classification BCE uses the known-label mask and the fixed `batch_size * n_labels` denominator from Eq. (11).
- View supervision additionally excludes unavailable views, so their zero placeholders do not become training examples. The multi-view objective averages over the total number of views.
- Orthogonality sums the squared cosine inner products over available views and divides by batch size, as in Eq. (6). It does not divide by the number of views again.
- Reconstruction first averages squared feature error within each view, then sums available views and divides by batch size, as in Eq. (7).
- The adaptive view-confidence network receives gradients through the paper objective. No extra quality-target loss is introduced.

Algorithm 1 mentions a `gamma` hyperparameter, but Eq. (12) and Appendix G specify only `alpha` and `beta` as objective coefficients. This version follows Eq. (12) and does not invent a `gamma` term.

## Validation confidence and growth

Confidence uses separate positive and negative averages from Eq. (1), computed across the whole validation split. Availability and known-label masks restrict the valid entries. Summing batch means would give small final batches disproportionate influence and would not implement the stated score.

When a view lacks either positive or negative valid validation labels, its balanced score is undefined; it must not be treated as a measured weak view. Expansion decisions use only valid view scores. Newly appended parameters are added to the optimizer. The stage counts form part of checkpoint metadata because network structure changes during training.

The default classifier-count cap is three total stages per view, including the base stage. If the weakest view is already at this cap, the current epoch adds no stage; it does not choose a different view instead. `--max-view-layers` exposes this resource-control setting, which is not a bound specified by Eqs. (1)–(5). Default and overridden settings are stored with each run.

## MOEDF and SCFS details

The paper states that MOEDF uses four experts but does not fully enumerate expert widths and gate layers. The implementation uses separately gated banks of four linear experts for consistent and specific representations, then another bank of four linear experts for their concatenated fusion. Each gate is `Linear -> ReLU -> Linear -> softmax`. The encoder follows the supplied source widths `d_v -> round(0.8*d_v) -> round(0.8*d_v) -> 1500 -> c`, with ReLU hidden activations and final batch normalization; the decoder reverses these widths. Each classifier stage uses `Linear(c,c) -> ReLU -> Linear(c,c) -> Sigmoid` before the bounded increment. These layer choices are executable architecture choices rather than uniquely recoverable details of the manuscript.

The Eq. (9) MLP is shared across views and scores each cumulative probability vector independently before normalization across available views. This is one concrete interpretation of `D({P^(v)})`; the manuscript does not fully specify its internal architecture. Orthogonality is enforced by the Eq. (6) penalty rather than a separate exact projection step, resolving Algorithm 1's informal wording against Section 3.2.

SCFS swaps a contiguous fragment of exactly `floor(eta * feature_dim)` coordinates. The implementation uses zero-based indexing and includes the last valid fragment start. This resolves the paper's informal inclusive indexing without swapping an extra coordinate. Donors are different observed samples from the same view; unavailable samples cannot donate. If fewer than two samples of a view are available, or the fragment length is zero, that view is returned unchanged. Original input tensors are not mutated and remain the Eq. (7) reconstruction target.

SCFS runs only in training mode. Missing-view masks continue to govern losses and fusion after augmentation.

## Data preprocessing and old results

The default standardizer is fit separately for each view using only available training samples, then reused on validation and test samples. This avoids incorporating held-out feature statistics. Missing views remain masked placeholders rather than observations used to estimate the standardizer.

The prior bundle used whole-dataset normalization. The compatibility setting `legacy-all`, when explicitly requested, preserves that preprocessing convention and should be recorded in any reported comparison. It does not restore the old algorithm or make old results representative of this revision.

The loader reads the pre-generated fold permutation and uses `ceil(n * train_ratio)` training samples, followed by `ceil(n * validation_ratio)` validation samples, with the remainder reserved for testing. It reads the supplied missingness masks rather than generating new ones. Training exposes only the observed labels. Validation and test use complete held-out labels, following the supplied source data protocol; validation labels are used for branch growth and checkpoint selection, while test metrics do not choose checkpoints.

Dataset versions, folds, seeds, missingness masks, preprocessing mode, and classifier growth history must all match before comparing numerical results with another run. Evaluation-only mode restores the dataset name, missingness ratios, train/validation ratios, normalization mode, and fold index from the saved training configuration. Model dimensions and dynamic stage counts are also restored and checked. `--data-root` permits relocation of the original files, while device, batch size, and worker count remain runtime settings. Equal dimensions do not prove that two dataset files are identical. Checkpoints without the required dataset metadata, including synthetic smoke-test checkpoints, are rejected by benchmark evaluation mode.

An explicit output directory must be nonexistent or empty; omitting it creates a unique timestamped run directory. The effective restored settings are written to `config.json` before evaluation.

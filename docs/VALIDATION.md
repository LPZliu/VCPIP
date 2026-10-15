# Release validation

These checks validate implementation behavior and data compatibility. No benchmark training, retraining, or reproduction of the manuscript's reported performance was performed.

## Environment

Checks were performed on Windows using Python 3.9.23, PyTorch 2.8.0+cu129, NumPy 2.0.2, SciPy 1.13.1, and scikit-learn 1.6.1. The model checks described below used the CPU. An installed CUDA-enabled PyTorch build does not imply that GPU execution was validated.

## Completed checks

| Check | Observed result | What it establishes |
| --- | --- | --- |
| Python compilation | Passed `compileall` | Source files parse successfully in the checked environment. |
| Command-line entry point | `python main.py --help` completed | The entry point and command-line definitions import and initialize. |
| Automated tests | 16 passed, 0 failed, 0 skipped | Formula reductions, masking, augmentation, growth, checkpoint, loader, and metric behaviors passed the cases in `tests/`. |
| Synthetic model/loss check | Finite objectives and gradients; original inputs unchanged | Forward/backward paths and SCFS reconstruction-target preservation work on the synthetic case. |
| Synthetic dynamic expansion | New stage registered with the optimizer and received gradients | Newly added classifiers participate in optimization. |
| Synthetic checkpoint round trip | Maximum absolute prediction difference `0.0` | Saving and restoring the dynamic architecture preserved predictions for the checked case. |
| Corel5k data compatibility | Four real samples, six view dimensions `[100, 1000, 512, 4096, 4096, 4096]`, 260 labels; finite CPU predictions | The loader and full model accept the available Corel5k feature format. This was inference only, with no optimizer updates. |
| Evaluation-metric parity | A random `24 × 8` prediction/label case agreed with the source routine to maximum absolute difference `1.1102230246251565e-16` | The rewritten metric computation matches the source on this controlled case. It does not establish parity for every possible input. |

The synthetic check exercises a small optimizer update to validate gradient and parameter-registration behavior. It does not train on a benchmark dataset. Real-data verification was restricted to loading and inference.

The automated tests include numerical checks of Eqs. (4)–(7) and (11)–(12), all-prefix supervision, detached residual targets, masked BCE gradients, SCFS donor and input-preservation rules, bounded incremental probabilities, Eqs. (9)–(10) and confidence-network gradients, stage registration, checkpoint prediction equality, and validation-confidence aggregation across batch partitions. Synthetic MATLAB cases check training-only normalization, held-out and missing-feature perturbation behavior, transposed inputs, and split membership. Metric cases cover perfect predictions, the inherited top-k convention, and zero-label handling.

## Repeatable commands

```bash
python -m compileall -q main.py model.py myloss.py MLdataset.py evaluation.py utils.py tests
python main.py --help
python main.py --smoke-test --device cpu
python -m unittest discover -s tests -v
```

Smoke-test output is written to a new timestamped directory under `records/`, including `smoke_results.json` and a synthetic checkpoint. The checkpoint is for implementation checks only and is rejected by benchmark evaluation mode.

## Interpretation

These checks do not verify convergence, accuracy, uncertainty estimates, multi-seed robustness, GPU equivalence, or any improvement over prior methods. The mathematical conventions listed in [implementation notes](IMPLEMENTATION_NOTES.md) remain relevant when designing a future benchmark run. Historical results from the previous experimental code are not results for this revision.

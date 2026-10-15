# Provenance and attribution

## Specification

The method specification was the supplied manuscript **View Confidence Perception-Driven Incremental Prediction for Incomplete Multi-view Multi-label Learning**, by Pingzhu Liu, Zunnan Xu, Chunming He, Zhirui Fang, Zitong Yu, Xiu Li, and Chengliang Liu. The method/code map refers to its numbered equations and sections.

The work is cited as a 2026 manuscript. A publication link can be added when available; the manuscript PDF is not included in this source release.

## Starting code

The implementation was prepared from the user-supplied local `RANK-main` development bundle. That bundle contained RANK attribution alongside multiple experimental VCPIP-related training and model files. Its README identifies the original work as [Reliable Representation Learning for Incomplete Multi-View Missing Multi-Label Classification](https://arxiv.org/pdf/2303.17117).

The following original citation is retained from that README:

```bibtex
@article{liu2025reliable,
  title={Reliable representation learning for incomplete multi-view missing multi-label classification},
  author={Liu, Chengliang and Wen, Jie and Xu, Yong and Zhang, Bob and Nie, Liqiang and Zhang, Min},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE},
  volume={47},
  number={6},
  pages={4940--4956}
}
```

The starting bundle informed data-file compatibility, evaluation routines, and encoder/decoder and expert implementation structure. The cleaned release keeps a single public entry point and a single active model/loss implementation. The algorithm was revised against the supplied manuscript; it is not merely a rename of an unchanged RANK implementation.

## Changes that affect numerical behavior

| Starting experimental code | This manuscript-aligned revision |
| --- | --- |
| Extra discriminator, positive-pair, and global-label-guidance terms in an active training objective | Four terms from Eq. (12), with `alpha` for orthogonality and `beta` for reconstruction |
| Supervision of selected current/previous states | All cumulative states plus latest residual supervision from Eq. (5) |
| Confidence computed through batch-level means without the separate Eq. (1) denominators | Dataset-level positive and negative means, using valid observed entries |
| Hard-coded expansion behavior and device assumptions | Configurable threshold, dynamic parameter registration, and device-aware creation |
| Prediction path different from Eq. (10) | Weighted probability/feature interaction followed by final regression |
| Whole-dataset feature normalization | Training-only observed-view normalization by default |

These changes mean that previous logs, trained weights, and reported metrics cannot be attributed to this revision. No benchmark retraining was carried out as part of the cleanup.

## Layout reference

[LMVCAT](https://github.com/justsmart/LMVCAT) was used as the reference for a compact research-repository layout: `main.py`, `model.py`, `myloss.py`, `MLdataset.py`, `evaluation.py`, `utils.py`, `data/`, and `records/`. Its transformer method is not presented as a VCPIP component, and its publication citation is not used as the citation for this work.

## Distribution and licensing

No license file was found in the supplied starting bundle. This cleanup does not invent an MIT, Apache, or other blanket license for material whose licensing terms have not been established. Source attribution is preserved; public repository visibility alone is not a license grant. Any future repository license should be chosen by the rights holders with the provenance of retained material taken into account.

Datasets, historical run outputs, trained checkpoints, the supplied manuscript PDF, and the original local backup are outside the public source release. The original source bundle is retained locally in a separate backup rather than committed as unused experimental files.

# Method-to-code map / 论文创新与代码对应

This document maps the supplied VCPIP manuscript to the cleaned implementation. It separates the paper's definitions from the conventions needed to make underspecified formulas executable. 本文档逐项对应论文贡献与实现；实现约定不作为论文原文已经规定的事实。

## 1. View-confidence-aware incremental prediction / 视图置信度感知增量预测

**Paper:** Section 3.1, Eqs. (1)–(5).

| Paper operation / 论文步骤 | Code location / 代码位置 | Behavior / 实现行为 |
| --- | --- | --- |
| Eq. (1): view confidence score | `main.py:compute_view_confidence`, `main.py:evaluate` | Sum the mean predicted probability on observed positive labels and the mean negative probability on observed negative labels; accumulate numerators and denominators across the complete validation split. 使用完整验证集的正、负标签各自计数归一化，不平均批次均值。 |
| Eqs. (2)–(3): relative performance and weakest view | `main.py:evaluate_and_add_layers` | Compare minimum and maximum valid scores; when their ratio is below `tau`, append one classifier to the weakest view if it has not reached the configured stage limit, and register its new parameters with the optimizer. 在验证后按最低置信度视图增加分类器；该视图达到配置上限时本轮不扩展。 |
| Eq. (4): positive residual target | `myloss.py:progressive_multiview_loss` | Build `Y * (1 - previous_cumulative_probability.detach())` for the latest stage. 目标来自前面所有阶段的累计概率，不只来自上一分类器；目标本身不反向传播。 |
| Eq. (5): dense supervision | `myloss.py:progressive_multiview_loss` | Supervise every cumulative state from stage 1 through the current stage, plus the latest residual prediction. 未扩展视图只使用基础 BCE。 |
| Dynamic architecture | `model.py:DynamicBoostingClassifier`, `VCPIP.add_classifier_layer`, `main.py:save_checkpoint`, `main.py:load_checkpoint` | Keep a base classifier plus zero or more auxiliary classifiers for each view; checkpoint metadata reconstructs the stage counts before loading weights. 各视图初始附加分支数量 `boost_counts` 为 0。 |

The manuscript calls each stage output a probability but adds these outputs before binary cross-entropy. Independent sigmoid outputs can sum above one. This implementation therefore uses bounded additive increments:

$$P_0=0,\quad p_t=(1-P_{t-1})\odot\sigma(h_t(Z)),\quad P_t=P_{t-1}+p_t.$$

This preserves the cumulative-sum form and makes its BCE inputs valid probabilities. **This parameterization is an explicit implementation convention; it is not specified in the manuscript.** 这一约定需要作者核对，不应写成原文唯一确定的实现。

## 2. Mixture-of-experts decouple-fusion / 混合专家解耦融合

**Paper:** Section 3.2, Eqs. (6)–(7).

| Component / 模块 | Code location / 代码位置 | Role / 功能 |
| --- | --- | --- |
| View-specific encoder $E_v$ | `model.py:Encoder` | Encode each view to its latent representation; availability masks control its contribution to losses and fusion. 将各视图特征编码到潜空间，并通过掩码限制其有效贡献。 |
| Decoupling component $\Phi_{dec}$ | `model.py`: `MoEDecouple` | Use separately gated expert banks to obtain consistent and specific features. 由专家加权得到一致特征与特有特征。 |
| Eq. (6): orthogonality | `myloss.py:orthogonality_loss` | Sum squared inner products of L2-normalized consistent/specific features over available views, then divide by batch size. 按论文对视图求和，不额外除以视图数。 |
| Fusion component $\Psi_{fuse}$ | `model.py`: `MoEFuser` | Fuse consistent and specific features through a gated expert bank to form $Z'^{(v)}$. |
| View-specific decoder $D_v$ | `model.py:Decoder` | Reconstruct each view from its fused representation. |
| Eq. (7): reconstruction | `myloss.py:reconstruction_loss` | Compute per-view feature-mean squared error against the original input, apply the availability mask, sum views, and average samples. 重建目标为交换前的原始输入。 |

The default number of experts is four, following Appendix G. Expert architecture and gating choices are recorded in [implementation notes](IMPLEMENTATION_NOTES.md).

## 3. Stochastic cross-sample fragment swapping / 随机跨样本片段交换

**Paper:** Section 3.3, Eq. (8).

`model.py:stochastic_cross_sample_fragment_swap` replaces a contiguous feature fragment of length `floor(eta * feature_dim)` using another observed sample from the same view and batch. Donors differ from recipients. Missing views cannot donate fragments. Validation and test inference do not apply SCFS.

`SCFS` 在同一视图、同一批次的可观测样本之间交换连续片段；供体与接收样本不同。只有一个可用样本或片段长度为 0 时保持输入不变。增强过程不原地修改输入，因此 Eq. (7) 的重建目标仍为原始特征。

## 4. Confidence-weighted classification / 置信度加权分类

**Paper:** Section 3.4, Eqs. (9)–(12).

| Formula | Code location | Correspondence |
| --- | --- | --- |
| Eq. (9) | `model.py:VCPIP.forward`, `model.py:masked_view_softmax` | A shared MLP evaluates each view's cumulative prediction; softmax normalizes weights over available views. 缺失视图权重为 0，正常样本可用视图权重和为 1。 |
| Eq. (10) | `model.py:VCPIP.forward` | Form the weighted sum of view probabilities and weighted sum of refined features separately; apply feature softmax, elementwise multiplication, ReLU, and the final sigmoid regression layer in that order. |
| Eq. (11) | `myloss.py:masked_binary_cross_entropy` | Mask unknown labels and divide the BCE sum by batch size × number of classes. 使用论文的固定分母，而非已知标签计数。 |
| Eq. (12) | `myloss.py:VCPIPLoss.forward` | `classification + multiview + alpha * orthogonality + beta * reconstruction`. |

Eq. (10) multiplies a label-space vector by a latent-space vector without defining a dimension-matching operator. The implementation sets the latent dimension equal to the number of labels. **This is a documented implementation choice, not a latent dimension given by the paper.** 论文未提供潜空间维数或匹配算子；当前采用 `latent_dim = n_labels`，避免额外引入未定义映射。

## 5. Boundaries / 版本与验证边界

- The old source's extra discriminator supervision, positive-pair loss, and global-label guidance are absent from Eq. (12) and are not included in this version's objective. 旧实验中这些项不能冒充本论文公式。
- Checkpoint loading must restore the number of auxiliary classifiers for every view. Old experiment checkpoints are not promised to be compatible with this model.
- Formula, shape, masking, gradient, and file-loading checks establish implementation behavior. They do not establish the manuscript's reported benchmark performance.
- This release does not retrain benchmark models. Paper result tables are not republished as results measured with this revision.

import torch
from torch import nn
from torch.nn import functional as F


def masked_binary_cross_entropy(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target and mask must have identical shapes")
    return (F.binary_cross_entropy(prediction, target, reduction="none") * mask).mean()


def progressive_multiview_loss(stage_probabilities, cumulative_probabilities,
                               labels, view_mask, label_mask):
    losses = []
    for v, (increments, prefixes) in enumerate(zip(stage_probabilities, cumulative_probabilities)):
        mask = label_mask * view_mask[:, v:v + 1]
        loss = sum(masked_binary_cross_entropy(p, labels, mask) for p in prefixes)
        if len(increments) > 1:
            residual_target = labels * (1.0 - prefixes[-2].detach())
            loss = loss + masked_binary_cross_entropy(increments[-1], residual_target, mask)
        losses.append(loss)
    return torch.stack(losses).mean()


def orthogonality_loss(shared, private, view_mask):
    terms = []
    for v, (consistent, specific) in enumerate(zip(shared, private)):
        cosine = (F.normalize(consistent, dim=-1) * F.normalize(specific, dim=-1)).sum(-1)
        terms.append((cosine.square() * view_mask[:, v]).mean())
    return torch.stack(terms).sum()


def reconstruction_loss(reconstructions, original_views, view_mask):
    terms = []
    for v, (prediction, original) in enumerate(zip(reconstructions, original_views)):
        error = (prediction - original).square().mean(dim=-1)
        terms.append((error * view_mask[:, v]).mean())
    return torch.stack(terms).sum()


class VCPIPLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("Loss weights must be non-negative")
        self.alpha, self.beta = alpha, beta

    def forward(self, outputs, original_views, labels, view_mask, label_mask):
        classification = masked_binary_cross_entropy(outputs["predictions"], labels, label_mask)
        multiview = progressive_multiview_loss(
            outputs["stage_probabilities"], outputs["cumulative_probabilities"],
            labels, view_mask, label_mask)
        orthogonality = orthogonality_loss(outputs["shared"], outputs["private"], view_mask)
        reconstruction = reconstruction_loss(outputs["reconstructions"], original_views, view_mask)
        total = classification + multiview + self.alpha * orthogonality + self.beta * reconstruction
        return dict(total=total, classification=classification, multiview=multiview,
                    orthogonality=orthogonality, reconstruction=reconstruction)

"""VCPIP architecture: VCA branches, MOEDF features, and SCFS augmentation.

This is a paper-guided implementation, not a checkpoint-compatible rewrite of
the earlier experimental ``new_model.py``. Two choices left unspecified by the
paper are explicit here: the embedding dimension equals the label count for
the elementwise product in Eq. (10), and additive classifier probabilities use
remaining-probability gates so that the cumulative predictions in Eq. (5) are
valid BCE inputs. Neither choice is presented as an additional contribution.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def stochastic_cross_sample_fragment_swap(
    views: Sequence[Tensor], availability: Tensor, eta: float = 0.2
) -> list[Tensor]:
    """SCFS, Eq. (8): replace one contiguous feature fragment per sample.

    Donors and recipients must both have the view, and a sample cannot donate
    to itself. Donors are sampled independently and may repeat. A view with
    fewer than two observed samples is unchanged. The caller's list and tensors
    remain untouched so Eq. (7) can reconstruct the original, clean inputs.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be between 0 and 1")
    augmented = list(views)
    for view_idx, features in enumerate(views):
        if features.ndim != 2:
            raise ValueError("SCFS expects each view to have shape [batch, features]")
        feature_count = features.shape[1]
        fragment_length = int(eta * feature_count)
        observed = torch.nonzero(
            availability[:, view_idx].to(device=features.device) > 0,
            as_tuple=False,
        ).flatten()
        sample_count = observed.numel()
        if fragment_length == 0 or sample_count < 2:
            continue

        # A nonzero cyclic offset samples uniformly from all other observed rows.
        recipient_positions = torch.arange(sample_count, device=features.device)
        offsets = torch.randint(1, sample_count, (sample_count,), device=features.device)
        donor_rows = observed[(recipient_positions + offsets) % sample_count]
        starts = torch.randint(
            0, feature_count - fragment_length + 1,
            (sample_count,), device=features.device,
        )
        feature_positions = torch.arange(feature_count, device=features.device)
        fragment_mask = (
            (feature_positions.unsqueeze(0) >= starts.unsqueeze(1))
            & (feature_positions.unsqueeze(0) < (starts + fragment_length).unsqueeze(1))
        )
        swapped = features.clone()
        swapped[observed] = torch.where(
            fragment_mask, features[donor_rows], features[observed]
        )
        augmented[view_idx] = swapped
    return augmented


class QualityNetwork(nn.Module):
    """Input-dependent expert routing used inside the MOEDF components."""

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_experts),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class MoEDecouple(nn.Module):
    """MOEDF, Section 3.2: independently routed shared and private experts.

    Orthogonality is enforced by the Eq. (6) loss, not by projecting features
    in this module. The four-expert default follows the paper's supplement.
    """

    def __init__(self, latent_dim: int, num_experts: int = 4):
        super().__init__()
        self.shared_experts = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(num_experts)]
        )
        self.private_experts = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(num_experts)]
        )
        self.shared_quality = QualityNetwork(latent_dim, num_experts)
        self.private_quality = QualityNetwork(latent_dim, num_experts)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        shared_weights = F.softmax(self.shared_quality(features), dim=-1)
        private_weights = F.softmax(self.private_quality(features), dim=-1)
        shared_values = torch.stack([expert(features) for expert in self.shared_experts], dim=1)
        private_values = torch.stack([expert(features) for expert in self.private_experts], dim=1)
        shared = (shared_weights.unsqueeze(-1) * shared_values).sum(dim=1)
        private = (private_weights.unsqueeze(-1) * private_values).sum(dim=1)
        return shared, private


class MoEFuser(nn.Module):
    """MOEDF fusion: expert-weighted reconstruction of a view representation."""

    def __init__(self, latent_dim: int, num_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(2 * latent_dim, latent_dim) for _ in range(num_experts)]
        )
        self.quality = QualityNetwork(2 * latent_dim, num_experts)

    def forward(self, shared: Tensor, private: Tensor) -> Tensor:
        features = torch.cat([shared, private], dim=-1)
        weights = F.softmax(self.quality(features), dim=-1)
        expert_values = torch.stack([expert(features) for expert in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_values).sum(dim=1)


class Encoder(nn.Module):
    """View encoder; hidden widths retain the supplied experiment's layout."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(1, round(0.8 * input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1500), nn.ReLU(),
            nn.Linear(1500, latent_dim), nn.BatchNorm1d(latent_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class Decoder(nn.Module):
    """Map the MOEDF representation back to the original view for Eq. (7)."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(1, round(0.8 * input_dim))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1500), nn.ReLU(),
            nn.Linear(1500, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class DynamicBoostingClassifier(nn.Module):
    """VCA stage outputs and their sums for Eqs. (4)-(5).

    Each stage applies a Linear/ReLU classifier and a Linear/Sigmoid probability
    readout to the original encoded view. The paper leaves the probability
    parameterization unspecified. We use p_1=q_1 and p_t=(1-P_{t-1})*q_t, where
    q_t is the stage sigmoid output. Therefore P_t=sum_{k<=t} p_k stays in [0,1].
    The Eq. (4) residual target is computed separately by the loss function.
    """

    def __init__(self, latent_dim: int, n_labels: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.stages = nn.ModuleList([self._make_stage()])

    def _make_stage(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.n_labels), nn.Sigmoid(),
        )

    def add_boost_layer(self) -> nn.Module:
        """Append a stage on the same device and dtype as the current branch."""
        reference = next(self.parameters())
        stage = self._make_stage().to(device=reference.device, dtype=reference.dtype)
        stage.train(self.training)
        self.stages.append(stage)
        return stage

    def forward(self, features: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        increments, cumulative = [], []
        total = torch.zeros(
            features.shape[0], self.n_labels, device=features.device, dtype=features.dtype
        )
        for stage in self.stages:
            increment = (1.0 - total) * stage(features)
            total = total + increment
            increments.append(increment)
            cumulative.append(total)
        return increments, cumulative


def masked_view_softmax(scores: Tensor, availability: Tensor) -> Tensor:
    """Eq. (9) over observed views; all-missing rows receive zero weights.

    Dataset construction should retain at least one observed view per sample.
    The all-missing fallback avoids NaNs and gives the final regression's prior.
    """
    observed = availability.to(device=scores.device) > 0
    masked_scores = scores.masked_fill(~observed, -torch.inf)
    all_missing = ~observed.any(dim=1, keepdim=True)
    safe_scores = torch.where(all_missing, torch.zeros_like(masked_scores), masked_scores)
    return F.softmax(safe_scores, dim=1) * observed.to(dtype=scores.dtype)


class VCPIP(nn.Module):
    """Connect SCFS, VCA, MOEDF and the weighted prediction in Eqs. (9)-(10).

    ``latent_dim == n_labels`` is an explicit implementation convention: Eq.
    (10) multiplies label probabilities and structural features elementwise but
    the paper supplies no intervening projection or embedding size. Missing
    views are excluded from Eq. (9)'s softmax as an availability convention.
    """

    def __init__(
        self, input_dims: Sequence[int], n_labels: int, num_experts: int = 4,
        latent_dim: int | None = None, boost_counts: Sequence[int] | None = None,
    ):
        super().__init__()
        self.input_dims = [int(dim) for dim in input_dims]
        self.n_labels = int(n_labels)
        self.latent_dim = self.n_labels if latent_dim is None else int(latent_dim)
        self.num_experts = int(num_experts)
        self.view_num = len(self.input_dims)
        if not self.input_dims or any(dim < 1 for dim in self.input_dims):
            raise ValueError("input_dims must contain positive view feature dimensions")
        if self.n_labels < 1 or self.num_experts < 1:
            raise ValueError("n_labels and num_experts must be positive")
        if self.latent_dim != self.n_labels:
            raise ValueError("Eq. (10) requires latent_dim == n_labels in this implementation")

        self.encoder_list = nn.ModuleList([
            Encoder(dim, self.latent_dim) for dim in self.input_dims
        ])
        self.decoder_list = nn.ModuleList([
            Decoder(dim, self.latent_dim) for dim in self.input_dims
        ])
        self.view_classifiers = nn.ModuleList([
            DynamicBoostingClassifier(self.latent_dim, self.n_labels)
            for _ in self.input_dims
        ])
        self.decouple_modules = nn.ModuleList([
            MoEDecouple(self.latent_dim, self.num_experts) for _ in self.input_dims
        ])
        self.fuser_modules = nn.ModuleList([
            MoEFuser(self.latent_dim, self.num_experts) for _ in self.input_dims
        ])
        self.discriminator = nn.Sequential(
            nn.Linear(self.n_labels, self.n_labels), nn.ReLU(), nn.Linear(self.n_labels, 1)
        )
        self.regression = nn.Linear(self.n_labels, self.n_labels)
        if boost_counts is not None:
            if len(boost_counts) != self.view_num:
                raise ValueError("boost_counts must contain one count per view")
            for view_idx, count in enumerate(boost_counts):
                if int(count) != count or count < 0:
                    raise ValueError("boost_counts must contain nonnegative integers")
                for _ in range(int(count)):
                    self.add_classifier_layer(view_idx)

    @property
    def boost_counts(self) -> list[int]:
        """Number of auxiliary stages per view; excludes the primary stage."""
        return [len(branch.stages) - 1 for branch in self.view_classifiers]

    def add_classifier_layer(self, view_idx: int) -> nn.Module:
        if not 0 <= view_idx < self.view_num:
            raise ValueError(f"Invalid view index: {view_idx}")
        return self.view_classifiers[view_idx].add_boost_layer()

    def get_view_layer_count(self, view_idx: int) -> int:
        if not 0 <= view_idx < self.view_num:
            raise ValueError(f"Invalid view index: {view_idx}")
        return len(self.view_classifiers[view_idx].stages)

    def forward(
        self, views: Sequence[Tensor], availability: Tensor, eta: float = 0.2
    ) -> dict:
        if len(views) != self.view_num:
            raise ValueError("The number of input views does not match input_dims")
        batch_size = views[0].shape[0]
        if tuple(availability.shape) != (batch_size, self.view_num):
            raise ValueError("availability must have shape [batch, number of views]")
        for features, input_dim in zip(views, self.input_dims):
            if features.ndim != 2 or tuple(features.shape) != (batch_size, input_dim):
                raise ValueError("Each view must have shape [batch, its input dimension]")

        inputs = (
            stochastic_cross_sample_fragment_swap(views, availability, eta)
            if self.training else views
        )
        encoded = [encoder(features) for encoder, features in zip(self.encoder_list, inputs)]
        shared, private, refined, reconstructions = [], [], [], []
        stage_probabilities, cumulative_probabilities, view_probabilities = [], [], []
        for view_idx, features in enumerate(encoded):
            shared_view, private_view = self.decouple_modules[view_idx](features)
            refined_view = self.fuser_modules[view_idx](shared_view, private_view)
            shared.append(shared_view)
            private.append(private_view)
            refined.append(refined_view)
            reconstructions.append(self.decoder_list[view_idx](refined_view))
            increments, cumulative = self.view_classifiers[view_idx](features)
            stage_probabilities.append(increments)
            cumulative_probabilities.append(cumulative)
            view_probabilities.append(cumulative[-1])

        stacked_predictions = torch.stack(view_probabilities, dim=1)
        scores = self.discriminator(stacked_predictions).squeeze(-1)
        view_weights = masked_view_softmax(scores, availability)
        weights = view_weights.unsqueeze(-1)
        semantic = (weights * stacked_predictions).sum(dim=1)
        structural = (weights * torch.stack(refined, dim=1)).sum(dim=1)
        # Eq. (10): weighted semantic probabilities times structural softmax.
        fused = semantic * F.softmax(structural, dim=-1)
        predictions = torch.sigmoid(self.regression(F.relu(fused)))
        return {
            "predictions": predictions,
            "reconstructions": reconstructions,
            "shared": shared,
            "private": private,
            "view_probabilities": view_probabilities,
            "stage_probabilities": stage_probabilities,
            "cumulative_probabilities": cumulative_probabilities,
            "view_weights": view_weights,
            "encoded": encoded,
            "refined": refined,
            "semantic": semantic,
            "structural": structural,
        }


def get_model(input_dims, n_labels, device="cpu", **kwargs) -> VCPIP:
    """Construct the paper-guided model on the requested device."""
    return VCPIP(input_dims, n_labels, **kwargs).to(device)

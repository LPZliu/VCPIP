"""Independent numeric and synthetic integration checks; no benchmark training."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (
    compute_view_confidence, evaluate_and_add_layers, load_checkpoint,
    save_checkpoint,
)
from model import VCPIP, stochastic_cross_sample_fragment_swap
from myloss import (
    VCPIPLoss, masked_binary_cross_entropy, orthogonality_loss,
    progressive_multiview_loss, reconstruction_loss,
)


def manual_bce(probabilities, targets, mask):
    """Scalar Python oracle with the manuscript's n*c denominator."""
    total = 0.0
    for i, row in enumerate(probabilities):
        for j, probability in enumerate(row):
            target = float(targets[i][j])
            total -= float(mask[i][j]) * (
                target * math.log(float(probability))
                + (1.0 - target) * math.log(1.0 - float(probability))
            )
    return total / (len(probabilities) * len(probabilities[0]))


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
        self.label_mask = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64)

    def test_eq4_eq5_residual_and_every_prefix(self):
        increments = [torch.full((2, 2), p, dtype=torch.float64) for p in (.2, .15, .1)]
        prefixes = [torch.full((2, 2), p, dtype=torch.float64) for p in (.2, .35, .45)]
        other = torch.full((2, 2), .6, dtype=torch.float64)
        view_mask = torch.tensor([[1., 1.], [1., 0.]], dtype=torch.float64)
        residual = [[.65, 0.], [0., .65]]
        expected_first = sum(manual_bce(p.tolist(), self.labels.tolist(), self.label_mask.tolist())
                             for p in prefixes)
        expected_first += manual_bce(increments[-1].tolist(), residual, self.label_mask.tolist())
        expected_second = manual_bce(other.tolist(), self.labels.tolist(), [[1., 0.], [0., 0.]])
        actual = progressive_multiview_loss(
            [increments, [other]], [prefixes, [other]], self.labels, view_mask, self.label_mask,
        )
        self.assertAlmostEqual(float(actual), (expected_first + expected_second) / 2, places=12)

    def test_eq4_residual_target_is_detached(self):
        increments = [torch.full((2, 2), p, dtype=torch.float64, requires_grad=True)
                      for p in (.2, .15, .1)]
        prefixes = [torch.full((2, 2), p, dtype=torch.float64, requires_grad=True)
                    for p in (.2, .35, .45)]
        loss = progressive_multiview_loss(
            [increments], [prefixes], self.labels,
            torch.ones((2, 1), dtype=torch.float64), self.label_mask,
        )
        loss.backward()
        # Each prefix is independent here: only its own BCE may differentiate it.
        expected_prefix_grad = ((.35 - self.labels) / (.35 * .65)) * self.label_mask / 4
        expected_increment_grad = ((.1 - self.labels * .65) / (.1 * .9)) * self.label_mask / 4
        torch.testing.assert_close(prefixes[1].grad, expected_prefix_grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(increments[-1].grad, expected_increment_grad, rtol=1e-12, atol=1e-12)

    def test_eq6_sums_views_instead_of_averaging(self):
        shared = [torch.tensor([[1., 0.], [1., 0.]])] * 2
        private = [torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[1., 0.], [1., 0.]])]
        mask = torch.tensor([[1., 1.], [1., 0.]])
        # View 1 contributes (1 + 0)/2; view 2 contributes (1 + masked 1)/2.
        self.assertEqual(float(orthogonality_loss(shared, private, mask)), 1.0)

    def test_eq7_uses_original_targets_feature_mean_and_view_mask(self):
        originals = [torch.tensor([[1., 3.], [2., 4.]]), torch.tensor([[0.], [2.]])]
        reconstructions = [torch.tensor([[3., 1.], [9., 9.]]), torch.tensor([[1.], [3.]])]
        mask = torch.tensor([[1., 1.], [0., 1.]])
        # ((4+4)/2 + masked error)/2 + (1+1)/2 = 3.
        self.assertEqual(float(reconstruction_loss(reconstructions, originals, mask)), 3.0)

    def test_eq11_denominator_and_completely_masked_gradients(self):
        probability = torch.tensor([[.8, .1], [.2, .7]], dtype=torch.float64, requires_grad=True)
        only_first = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
        actual = masked_binary_cross_entropy(probability, self.labels, only_first)
        self.assertAlmostEqual(float(actual.detach()), -math.log(.8) / 4, places=12)
        zero_mask = torch.zeros_like(self.labels)
        loss = masked_binary_cross_entropy(probability, self.labels, zero_mask)
        loss += progressive_multiview_loss(
            [[probability]], [[probability]], self.labels,
            torch.zeros((2, 1), dtype=torch.float64), torch.ones_like(self.labels),
        )
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(int(torch.count_nonzero(probability.grad)), 0)

    def test_eq12_exact_composition(self):
        outputs = {
            "predictions": torch.full((2, 2), .6, dtype=torch.float64),
            "stage_probabilities": [[torch.full((2, 2), .4, dtype=torch.float64)]],
            "cumulative_probabilities": [[torch.full((2, 2), .4, dtype=torch.float64)]],
            "shared": [torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)],
            "private": [torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)],
            "reconstructions": [torch.tensor([[2., 2.], [2., 2.]], dtype=torch.float64)],
        }
        losses = VCPIPLoss(alpha=.7, beta=.3)(
            outputs, [torch.zeros((2, 2), dtype=torch.float64)], self.labels,
            torch.ones((2, 1), dtype=torch.float64), self.label_mask,
        )
        expected_class = manual_bce([[.6, .6], [.6, .6]], self.labels.tolist(), self.label_mask.tolist())
        expected_view = manual_bce([[.4, .4], [.4, .4]], self.labels.tolist(), self.label_mask.tolist())
        expected = expected_class + expected_view + .7 * .5 + .3 * 4
        self.assertAlmostEqual(float(losses["total"]), expected, places=12)
        self.assertEqual(set(losses), {"total", "classification", "multiview", "orthogonality", "reconstruction"})


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(73)

    def test_scfs_contiguity_availability_self_exclusion_and_no_mutation(self):
        views = [torch.arange(4.).view(4, 1).expand(4, 10).clone()]
        before = views[0].clone()
        original_object = views[0]
        mask = torch.tensor([[1.], [0.], [1.], [1.]])
        result = stochastic_cross_sample_fragment_swap(views, mask, eta=.3)
        self.assertIs(views[0], original_object)
        self.assertTrue(torch.equal(views[0], before))
        self.assertTrue(torch.equal(result[0][1], before[1]))
        for row in (0, 2, 3):
            changed = torch.nonzero(result[0][row] != before[row]).flatten()
            self.assertEqual(changed.numel(), 3)
            self.assertEqual(changed[-1] - changed[0], 2)
            self.assertIn(float(result[0][row, changed[0]]), {0., 2., 3.} - {float(row)})
        only_one = torch.tensor([[0.], [1.], [0.], [0.]])
        self.assertTrue(torch.equal(stochastic_cross_sample_fragment_swap(views, only_one, 1.)[0], before))
        self.assertTrue(torch.equal(stochastic_cross_sample_fragment_swap(views, mask, 0.)[0], before))

    def test_eq9_eq10_bounded_stages_and_confidence_gradient(self):
        model = VCPIP([5, 7], 3, num_experts=2, boost_counts=[2, 1]).double().eval()
        views = [torch.randn(4, 5, dtype=torch.float64), torch.randn(4, 7, dtype=torch.float64)]
        mask = torch.tensor([[1., 1.], [1., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
        output = model(views, mask)
        probabilities = torch.stack(output["view_probabilities"], dim=1)
        raw_scores = model.discriminator(probabilities).squeeze(-1)
        expected_weights = torch.zeros_like(raw_scores)
        expected_weights[0] = torch.softmax(raw_scores[0], dim=0)
        expected_weights[1, 0] = expected_weights[2, 1] = 1.
        semantic = (expected_weights.unsqueeze(-1) * probabilities).sum(dim=1)
        structural = (expected_weights.unsqueeze(-1) * torch.stack(output["refined"], dim=1)).sum(dim=1)
        expected_prediction = torch.sigmoid(model.regression(torch.relu(
            semantic * torch.softmax(structural, dim=1)
        )))
        torch.testing.assert_close(output["view_weights"], expected_weights, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(output["predictions"], expected_prediction, rtol=1e-12, atol=1e-12)
        for increments, prefixes in zip(output["stage_probabilities"], output["cumulative_probabilities"]):
            for stage, prefix in enumerate(prefixes):
                torch.testing.assert_close(prefix, sum(increments[:stage + 1]))
                self.assertTrue(bool(((prefix >= 0) & (prefix <= 1)).all()))
        output["predictions"].sum().backward()
        self.assertGreater(float(model.discriminator[0].weight.grad.abs().sum()), 0.)

    def test_dynamic_growth_registers_optimizer_parameters(self):
        model = VCPIP([4, 5], 3, num_experts=2).double().eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        old_ids = {id(parameter) for parameter in model.parameters()}
        event = evaluate_and_add_layers(model, {"confidence": [1.8, .5]}, optimizer)
        self.assertTrue(event["expanded"])
        self.assertEqual(model.boost_counts, [0, 1])
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in old_ids]
        self.assertTrue(new_parameters)
        self.assertTrue(all(id(parameter) in optimizer_ids for parameter in new_parameters))
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in new_parameters))
        self.assertFalse(model.view_classifiers[1].stages[-1].training)

    def test_checkpoint_restores_dynamic_topology_and_exact_predictions(self):
        config = {"input_dims": [4, 5], "n_labels": 3, "num_experts": 2}
        model = VCPIP(**config, boost_counts=[2, 1]).eval()
        views = [torch.randn(4, 4), torch.randn(4, 5)]
        mask = torch.tensor([[1., 1.], [1., 0.], [0., 1.], [1., 1.]])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            save_checkpoint(checkpoint_path, model, config, epoch=7, train_config={"fold_index": 2})
            restored, checkpoint = load_checkpoint(checkpoint_path)
            restored.eval()
            self.assertEqual(checkpoint["epoch"], 7)
            self.assertEqual(checkpoint["train_config"]["fold_index"], 2)
            self.assertEqual(restored.boost_counts, [2, 1])
            with torch.no_grad():
                torch.testing.assert_close(model(views, mask)["predictions"],
                                           restored(views, mask)["predictions"], rtol=0, atol=0)


class ConfidenceTests(unittest.TestCase):
    def test_balanced_vcs_uses_global_class_counts_across_unequal_batches(self):
        labels = torch.tensor([[1., 0.], [1., 1.], [0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
        view_mask = torch.tensor([[1., 0.], [1., 1.], [1., 1.], [0., 1.], [1., 1.]], dtype=torch.float64)
        label_mask = torch.tensor([[1., 1.], [1., 0.], [1., 1.], [1., 1.], [0., 1.]], dtype=torch.float64)
        probabilities = [torch.tensor([[.9, .2], [.8, .7], [.4, .1], [.6, .5], [.3, .9]], dtype=torch.float64),
                         torch.tensor([[.2, .8], [.7, .4], [.1, .3], [.8, .2], [.9, .6]], dtype=torch.float64)]
        expected = np.zeros((2, 4))
        for view in range(2):
            for sample in range(5):
                for label in range(2):
                    if view_mask[sample, view] and label_mask[sample, label]:
                        if labels[sample, label]:
                            expected[view, 0] += float(probabilities[view][sample, label])
                            expected[view, 1] += 1
                        else:
                            expected[view, 2] += 1 - float(probabilities[view][sample, label])
                            expected[view, 3] += 1
        full = compute_view_confidence(probabilities, labels, view_mask, label_mask)
        accumulated = torch.zeros((2, 4), dtype=torch.float64)
        for start, stop in ((0, 1), (1, 4), (4, 5)):
            accumulated += compute_view_confidence(
                [p[start:stop] for p in probabilities], labels[start:stop],
                view_mask[start:stop], label_mask[start:stop],
            )
        np.testing.assert_allclose(full.numpy(), expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(accumulated, full, rtol=0, atol=1e-12)
        expected_scores = expected[:, 0] / expected[:, 1] + expected[:, 2] / expected[:, 3]
        actual_scores = accumulated[:, 0] / accumulated[:, 1] + accumulated[:, 2] / accumulated[:, 3]
        np.testing.assert_allclose(actual_scores.numpy(), expected_scores, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

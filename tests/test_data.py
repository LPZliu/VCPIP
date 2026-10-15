"""Synthetic MATLAB split/normalization checks and independent metric oracles."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation
from MLdataset import IncDataset, load_incomplete_fold


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.order = np.r_[np.arange(5, 20), np.arange(5)]
        rows = np.arange(20, dtype=np.float32)
        self.views = [np.column_stack([rows, rows * 3 + 2]),
                      np.column_stack([rows * 2, rows ** 2, rows + 4])]
        self.labels = np.column_stack([rows % 2, (rows // 2) % 2, rows % 3 == 0]).astype(np.float32)
        self.view_mask = np.ones((20, 2), dtype=np.float32)
        self.view_mask[self.order[[1, 4]], 0] = 0
        self.view_mask[self.order[2], 1] = 0
        self.label_mask = np.ones((20, 3), dtype=np.float32)
        self.label_mask[self.order[[0, 15, 18]], :] = 0
        self.fold_path = self.root / "fold.mat"
        fold_contents = {}
        for name, value in (("folds_sample_index", (self.order + 1).reshape(-1, 1)),
                            ("folds_data", self.view_mask), ("folds_label", self.label_mask)):
            cell = np.empty((1, 1), dtype=object)
            cell[0, 0] = value
            fold_contents[name] = cell
        savemat(self.fold_path, fold_contents)

    def save_data(self, name, views=None, transpose=False):
        cell = np.empty((1, 2), dtype=object)
        for index, view in enumerate(self.views if views is None else views):
            cell[0, index] = view.T if transpose else view
        path = self.root / name
        savemat(path, {"X": cell, "label": self.labels.T if transpose else self.labels})
        return path

    def test_training_only_scaling_is_invariant_to_held_out_and_unobserved_values(self):
        path = self.save_data("original.mat")
        normalized, _, masks, _, order = load_incomplete_fold(path, self.fold_path)
        changed = [view.copy() for view in self.views]
        for view_index, view in enumerate(changed):
            view[self.order[14:]] += 100000.
            view[self.view_mask[:, view_index] == 0] -= 200000.
        changed_path = self.save_data("changed.mat", changed)
        normalized_changed = load_incomplete_fold(changed_path, self.fold_path)[0]
        for view_index, original in enumerate(self.views):
            observed_training = self.view_mask[self.order[:14], view_index].astype(bool)
            fit = original[self.order[:14]][observed_training].astype(np.float64)
            expected = ((original[order] - fit.mean(axis=0)) / fit.std(axis=0)) * masks[:, view_index:view_index + 1]
            np.testing.assert_allclose(normalized[view_index], expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(normalized[view_index][:14], normalized_changed[view_index][:14])
            self.assertTrue(np.all(normalized[view_index][masks[:, view_index] == 0] == 0))

    def test_transposed_feature_and_label_orientation_loads_identically(self):
        normal = load_incomplete_fold(self.save_data("normal.mat"), self.fold_path)
        transposed = load_incomplete_fold(self.save_data("transposed.mat", transpose=True), self.fold_path)
        for first, second in zip(normal[0], transposed[0]):
            np.testing.assert_array_equal(first, second)
        for index in range(1, 5):
            np.testing.assert_array_equal(normal[index], transposed[index])

    def test_split_sizes_permutation_and_training_label_visibility(self):
        path = self.save_data("dataset.mat")
        datasets = [IncDataset(path, self.fold_path, mode=mode) for mode in ("train", "val", "test")]
        self.assertEqual([len(dataset) for dataset in datasets], [14, 3, 3])
        np.testing.assert_array_equal(np.concatenate([d.sample_indices for d in datasets]), self.order)
        np.testing.assert_array_equal(datasets[0].cur_labels, self.labels[self.order[:14]] * self.label_mask[self.order[:14]])
        np.testing.assert_array_equal(datasets[1].cur_labels, self.labels[self.order[14:17]])
        np.testing.assert_array_equal(datasets[2].cur_labels, self.labels[self.order[17:]])
        self.assertTrue(np.all(datasets[1].cur_inc_L_ind == 1))
        self.assertTrue(np.all(datasets[2].cur_inc_L_ind == 1))


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_and_legacy_top_k_auc_definition(self):
        labels = np.eye(3, dtype=np.float64)
        probabilities = .1 + .8 * labels
        np.testing.assert_allclose(evaluation.do_metric(probabilities, labels), np.ones(9))
        probabilities = np.array([[.7, .2, .6, .1], [.3, .8, .4, .9], [.9, .6, .1, .5]])
        labels = np.array([[1, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0]])
        positives, negatives = labels.sum(), labels.size - labels.sum()
        tpr, fpr = [], []
        for count in range(1, labels.shape[1] + 1):
            true_positive = false_positive = 0
            for row in range(len(labels)):
                predicted_indices = sorted(range(labels.shape[1]), key=lambda j: probabilities[row, j], reverse=True)[:count]
                true_positive += sum(labels[row, j] == 1 for j in predicted_indices)
                false_positive += sum(labels[row, j] == 0 for j in predicted_indices)
            tpr.append(true_positive / positives)
            fpr.append(false_positive / negatives)
        expected_area = sum((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
                            for i in range(1, len(tpr))) / (fpr[-1] - fpr[0])
        self.assertAlmostEqual(evaluation.mlc_auc(probabilities, labels), expected_area, places=12)

    def test_all_zero_labels_have_documented_undefined_auc(self):
        result = evaluation.do_metric(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertTrue(np.isnan(result[3]))
        self.assertTrue(np.isnan(result[6]))
        np.testing.assert_array_equal(result[[0, 1, 2, 4, 5, 7, 8]], [0., 1., 1., 0., 1., 0., 0.])


if __name__ == "__main__":
    unittest.main()

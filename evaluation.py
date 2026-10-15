import numpy as np
from sklearn import metrics

METRIC_NAMES = ["AP", "1-HL", "1-RL", "AUC", "1-OE", "1-Coverage", "macro-AUC", "macro-F1", "micro-F1"]

def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank

def compute_one_error(y_prob, label):
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss += label[i, pos] < 0.5
    return loss * 1.0 / num_samples

def compute_coverage(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = 0
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage += max(rank[i, label[i, :] > 0.5]) - 1
    coverage = coverage * 1.0 / num_samples
    return coverage / num_labels

def compute_average_precision(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples

def mlc_auc(probabilities, labels):
    n, c = labels.shape
    positives = labels.sum()
    negatives = labels.size - positives
    if c < 2 or positives == 0 or negatives == 0:
        return float("nan")
    order = probabilities.argsort(axis=1)[:, ::-1]
    sorted_labels = np.take_along_axis(labels, order, axis=1)
    tp = sorted_labels.cumsum(axis=1).sum(axis=0)
    fp = n * np.arange(1, c + 1) - tp
    tpr, fpr = tp / positives, fp / negatives
    width = fpr[-1] - fpr[0]
    if width <= 0:
        return float("nan")
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * 0.5) / width)


def do_metric(y_prob, label):
    y_prob, label = np.asarray(y_prob), np.asarray(label)
    if y_prob.ndim != 2 or y_prob.shape != label.shape or not y_prob.size:
        raise ValueError("Expected matching nonempty sample-by-label matrices")
    if not np.isfinite(y_prob).all() or not np.isin(label, (0, 1)).all():
        raise ValueError("Predictions must be finite; labels must be binary")
    prediction = y_prob > 0.5
    valid = (label.sum(axis=0) > 0) & (label.sum(axis=0) < len(label))
    macro_auc = (float(np.mean([metrics.roc_auc_score(label[:, j], y_prob[:, j])
                              for j in np.flatnonzero(valid)])) if valid.any() else float("nan"))
    return np.array([
        compute_average_precision(y_prob, label),
        1 - np.mean(prediction != label),
        1 - metrics.label_ranking_loss(label, y_prob),
        mlc_auc(y_prob, label),
        1 - compute_one_error(y_prob, label),
        1 - compute_coverage(y_prob, label),
        macro_auc,
        metrics.f1_score(label, prediction, average="macro", zero_division=0),
        metrics.f1_score(label, prediction, average="micro", zero_division=0),
    ], dtype=float)

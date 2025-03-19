import inspect

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score as ras

from performance_evaluator.struct_ import Metric

ROUND = 4
AVAILABLE_METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "pr_auc_score": "PR AUC Score",
    "roc_auc_score": "ROC AUC Score",
    "f1_score": "F1-Score",
    "npv": "Negative Predictive Value",
    "fnr": "False Negative Rate",
    "fpr": "False Positive Rate",
    "fdr": "False Discovery Rate",
    "for_": "False Omission Rate",
    "plikelihood_ratio": "Positive Likelihood Ratio",
    "nlikelihood_ratio": "Negative Likelihood Ration",
    "prevalence_threshold": "Prevalence Threshold",
    "threat_score": "Threat Score",
    "mcc": "Mathews Correlation Coefficient",
}
MOST_REQUIRED = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Sensitivity",
    "specificity": "Specificity",
    "f1_score": "FScore",
    "mcc": "MCC",
}


def fn_exec_kwargs(fn, kwargs):
    keys = [k for k in inspect.getfullargspec(fn).args if k in kwargs]
    return fn(**{k: kwargs[k] for k in keys})


def format_(metric, name):
    fmt = "{:." + str(ROUND) + "f}"
    if name == "accuracy":
        overall = fmt.format(metric.sum())
    else:
        overall = fmt.format(metric.mean())
    return overall, [fmt.format(v_) for v_ in metric]


def confusion_matrix(actual, predicted, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(actual)):
        cm[actual[i]][predicted[i]] += 1
    return cm


def get_tp_fn_fp_tn(cm):
    tp = np.diagonal(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    return tp, fn, fp, tn


def accuracy(tp, fn, fp, tn):
    return (tp + tn) / (tp + fn + fp + tn)


def balanced_accuracy(tp, fn, fp, tn):
    return (recall(tp, fn) + specificity(tn, fp)) / 2


def precision(tp, fp):
    try:
        return tp / (tp + fp)
    except RuntimeWarning:
        return np.nan


def recall(tp, fn):
    return tp / (tp + fn)


def specificity(tn, fp):
    return tn / (tn + fp)


def f1_score(tp, fn, fp):
    ppv_ = precision(tp, fp)
    tpr_ = recall(tp, fn)
    return 2 * ((ppv_ * tpr_) / (ppv_ + tpr_))


def __pr_roc_auc_score(actual, probability, f):
    actual_onehot = np.eye(probability.shape[1])[actual]
    scores = []
    for i in range(probability.shape[1]):
        y_true = actual_onehot[:, i]
        y_score = probability[:, i]
        if f == "pr":
            score = average_precision_score(y_true, y_score)
        else:
            score = ras(y_true, y_score)
        scores.append(score)
    return np.array(scores)


def pr_auc_score(actual, probability):
    return __pr_roc_auc_score(actual, probability, f="pr")


def roc_auc_score(actual, probability):
    return __pr_roc_auc_score(actual, probability, f="roc")


def npv(tn, fn):
    return tn / (tn + fn)


def fnr(tp, fn):
    return 1 - recall(tp, fn)


def fpr(tn, fp):
    return 1 - specificity(tn, fp)


def fdr(tp, fp):
    return 1 - precision(tp, fp)


def for_(tn, fn):
    return 1 - npv(tn, fn)


def plikelihood_ratio(tp, fn, fp, tn):
    return recall(tp, fn) / fpr(tn, fp)


def nlikelihood_ratio(tp, fn, fp, tn):
    return fnr(tp, fn) / specificity(tn, fp)


def prevalence_threshold(tn, fp, fn, tp):
    return np.sqrt(fpr(tn, fp)) / (np.sqrt(recall(tp, fn)) + np.sqrt(fpr(tn, fp)))


def threat_score(tp, fn, fp):
    return tp / (tp + fn + fp)


def mcc(tp, fn, fp, tn):
    n = tn + tp + fn + fp
    s = (tp + fn) / n
    p = (tp + fp) / n
    return ((tp / n) - (s * p)) / np.sqrt(p * s * (1 - s) * (1 - p))


def evaluate(actual, predicted, probability, classes, required_metrics=None):
    if required_metrics is None:
        required_metrics = MOST_REQUIRED
    cm = confusion_matrix(actual, predicted, len(classes))
    tp, fn, fp, tn = get_tp_fn_fp_tn(cm)
    kwargs = {
        "actual": actual,
        "predicted": predicted,
        "probability": probability,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }
    overall_data = {}
    class_data = {}
    for m in required_metrics:
        metric = fn_exec_kwargs(globals()[m], kwargs)
        overall, class_ = format_(metric, m)
        overall_data[required_metrics[m]] = [overall]
        class_data[required_metrics[m]] = class_

    overall_df = pd.DataFrame.from_dict(overall_data).T
    idx = overall_df.index.values
    overall_df.insert(0, "Metrics", idx)
    overall_df.reset_index(drop=True, inplace=True)
    overall_df.columns = ["Metrics", "Values"]
    class_df = pd.DataFrame.from_dict(class_data)
    class_df.insert(0, "Class", classes)
    return Metric(overall_df, class_df)

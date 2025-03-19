import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from matplotlib.ticker import NullFormatter
from sklearn.metrics import (
    precision_recall_curve as pr_curve,
    average_precision_score as ap_score,
    roc_curve as roc_c,
    roc_auc_score as roc_auc_s
)

from performance_evaluator.config import CURRENT_CMAP
from performance_evaluator.metrics import confusion_matrix as conf_mat


def get_ax(ax):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    return ax


def set_common(ax, kwargs, CONFIG):
    if 'annot_kws' not in kwargs:
        ax.set_xticklabels([round(v, 1) for v in ax.get_xticks()],
                           fontdict=kwargs.get('ticklabels_fontdict', CONFIG['ticklabels_fontdict']))
        ax.set_yticklabels([round(v, 1) for v in ax.get_yticks()],
                           fontdict=kwargs.get('ticklabels_fontdict', CONFIG['ticklabels_fontdict']))
    if kwargs.get('title', CONFIG['title']):
        ax.set_title(kwargs.get('title', CONFIG['title']),
                     pad=kwargs.get('titlepad', CONFIG['titlepad']),
                     fontdict=kwargs.get('title_fontdict', CONFIG['title_fontdict']))
    if kwargs.get('xlabel', CONFIG['xlabel']):
        ax.set_xlabel(kwargs.get('xlabel', CONFIG['xlabel']),
                      labelpad=kwargs.get('xylabelpad', CONFIG['xylabelpad']),
                      fontdict=kwargs.get('xylabel_fontdict', CONFIG['xylabel_fontdict']))
    if kwargs.get('ylabel', CONFIG['ylabel']):
        ax.set_ylabel(kwargs.get('ylabel', CONFIG['ylabel']),
                      labelpad=kwargs.get('xylabelpad', CONFIG['xylabelpad']),
                      fontdict=kwargs.get('xylabel_fontdict', CONFIG['xylabel_fontdict']))
    return ax


def confusion_matrix(actual, predicted, classes, ax=None, **kwargs):
    from performance_evaluator.config import CONFUSION_MATRIX as CONFIG
    cm = conf_mat(actual, predicted, n_classes=len(classes))
    ax = get_ax(ax)
    sbn.heatmap(
        cm,
        ax=ax,
        cmap=kwargs.get('cmap', CONFIG['cmap']),
        annot=True,
        annot_kws=kwargs.get('annot_kws', CONFIG['annot_kws']),
        cbar=kwargs.get('cbar', CONFIG['cbar']),
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        fmt='d',
    )
    if kwargs.get('cbar', CONFIG['cbar']):
        cbar = ax.collections[0].colorbar
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
                                fontdict=kwargs.get('cbar_ticklabels_fontdict', CONFIG['cbar_ticklabels_fontdict']))
    set_common(ax, kwargs, CONFIG)
    ax.set_xticklabels(classes, rotation=kwargs.get('xticklabels_rotation', CONFIG['xticklabels_rotation']),
                       fontdict=kwargs.get('ticklabels_fontdict', CONFIG['ticklabels_fontdict']))
    ax.set_yticklabels(classes, rotation=kwargs.get('yticklabels_rotation', CONFIG['yticklabels_rotation']),
                       fontdict=kwargs.get('ticklabels_fontdict', CONFIG['ticklabels_fontdict']))
    plt.tight_layout()
    if kwargs.get('show', False):
        plt.show()


def precision_recall_curve(actual, probability, classes, ax=None, **kwargs):
    from performance_evaluator.config import PR_CURVE as CONFIG
    ax = get_ax(ax)
    actual_onehot = np.eye(probability.shape[1])[actual]
    for i in range(probability.shape[1]):
        y_true = actual_onehot[:, i]
        y_score = probability[:, i]
        precision, recall, _ = pr_curve(y_true, y_score)
        ap_score_ = ap_score(y_true, y_score)
        color = plt.get_cmap(CURRENT_CMAP)(float(i) / len(classes))
        ax.plot(recall, precision, label='{0} (AP={1})'.format(classes[i], round(ap_score_, 4)),
                color=color)
    ax = set_common(ax, kwargs, CONFIG)
    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.5, linestyle='dotted')
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_on()
    ax.tick_params(which='minor', length=0)
    ax.legend(prop=kwargs.get('legend_fontdict', CONFIG['legend_fontdict']),
              ncol=kwargs.get('legend_ncol', CONFIG['legend_ncol']))
    plt.tight_layout()
    if kwargs.get('show', False):
        plt.show()


def roc_curve(actual, probability, classes, ax=None, **kwargs):
    from performance_evaluator.config import ROC_CURVE as CONFIG
    ax = get_ax(ax)
    actual_onehot = np.eye(probability.shape[1])[actual]
    for i in range(probability.shape[1]):
        y_true = actual_onehot[:, i]
        y_score = probability[:, i]
        fpr, tpr, _ = roc_c(y_true, y_score)
        auc_score = roc_auc_s(y_true, y_score)
        color = plt.get_cmap(CURRENT_CMAP)(float(i) / len(classes))
        ax.plot(fpr, tpr, label='{0} (AUC={1})'.format(classes[i], round(auc_score, 4)),
                color=color)
    ax = set_common(ax, kwargs, CONFIG)
    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.5, linestyle='dotted')
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_on()
    ax.tick_params(which='minor', length=0)
    ax.legend(prop=kwargs.get('legend_fontdict', CONFIG['legend_fontdict']),
              ncol=kwargs.get('legend_ncol', CONFIG['legend_ncol']))
    plt.tight_layout()
    if kwargs.get('show', False):
        plt.show()

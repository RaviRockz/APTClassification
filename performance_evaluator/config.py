import os

from matplotlib import pyplot as plt

root_path = os.path.dirname(__file__)

# Default
COLOR_MAPS = plt.colormaps()
CURRENT_CMAP = "viridis"

# Confusion Matrix
CONFUSION_MATRIX = dict(
    cmap=CURRENT_CMAP,
    cbar=False,
    cbar_ticklabels_fontdict=dict(font="IBM Plex Mono", size=13),
    annot_kws=dict(font="IBM Plex Mono", size=13),
    xticklabels_rotation=0,
    yticklabels_rotation=90,
    ticklabels_fontdict=dict(font="IBM Plex Mono", size=13),
    title="Confusion Matrix",
    titlepad=10,
    title_fontdict=dict(font="IBM Plex Mono", size=13),
    xlabel="Predicted Class",
    ylabel="Actual Class",
    xylabelpad=10,
    xylabel_fontdict=dict(font="IBM Plex Mono", size=13),
)

# Precision Recall Curve
PR_CURVE = dict(
    ticklabels_fontdict=dict(font="IBM Plex Mono", size=13),
    title="Precision-Recall Curve",
    titlepad=10,
    title_fontdict=dict(font="IBM Plex Mono", size=13),
    xlabel="Recall",
    ylabel="Precision",
    xylabelpad=10,
    xylabel_fontdict=dict(font="IBM Plex Mono", size=13),
    legend_fontdict=dict(family="IBM Plex Mono", size=13),
    legend_ncol=1,
)

# Receiver Operating Characteristic Curve
ROC_CURVE = dict(
    ticklabels_fontdict=dict(font="IBM Plex Mono", size=12),
    title="Receiver Operating Characteristic Curve",
    titlepad=15,
    title_fontdict=dict(font="IBM Plex Mono", size=16),
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    xylabelpad=15,
    xylabel_fontdict=dict(font="IBM Plex Mono", size=14),
    legend_fontdict=dict(family="IBM Plex Mono", size=12),
    legend_ncol=1,
)

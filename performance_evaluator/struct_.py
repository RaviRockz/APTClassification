import numpy as np


class Prediction:
    def __init__(self, actual=None, predicted=None, probability=None, classes=None):
        self.actual = actual
        self.predicted = predicted
        self.probability = probability
        self.classes = list(range(len(np.unique(actual)))) if classes is None else classes


class EpochPrediction:
    def __init__(self):
        self.prediction = []


class Metric:
    def __init__(self, om, cm):
        self.overall_metrics = om
        self.class_metrics = cm

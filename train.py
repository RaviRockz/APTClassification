if True:
    from reset_random import reset_random

    reset_random()
import os
import shutil

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_handler import load_data
from model import get_ensemble_model
from utils import CLASSES, TrainingCallback, plot
from prepare_data import CLASSES as classes

plt.rcParams["font.family"] = "IBM Plex Mono"

ACC_PLOT = plt.figure(num=2, figsize=(4.8, 3.6))
LOSS_PLOT = plt.figure(num=3, figsize=(4.8, 3.6))


RESULTS_PLOT = {
    "80:20": {
        "Train": {
            "CONF_MAT": plt.figure(figsize=(10, 10), num=4),
            "PR_CURVE": plt.figure(figsize=(10, 10), num=5),
            "ROC_CURVE": plt.figure(figsize=(10, 10), num=6),
        },
        "Test": {
            "CONF_MAT": plt.figure(figsize=(10, 10), num=7),
            "PR_CURVE": plt.figure(figsize=(10, 10), num=8),
            "ROC_CURVE": plt.figure(figsize=(10, 10), num=9),
        },
    },
    "70:30": {
        "Train": {
            "CONF_MAT": plt.figure(figsize=(10, 10), num=10),
            "PR_CURVE": plt.figure(figsize=(10, 10), num=11),
            "ROC_CURVE": plt.figure(figsize=(10, 10), num=12),
        },
        "Test": {
            "CONF_MAT": plt.figure(figsize=(10, 10), num=13),
            "PR_CURVE": plt.figure(figsize=(10, 10), num=14),
            "ROC_CURVE": plt.figure(figsize=(10, 10), num=15),
        },
    },
}


def get_data():
    df = load_data()
    df['Class'].replace({v: k for k, v in enumerate(classes)}, inplace=True)
    x_, y_ = df.values[:, :-1], df.values[:, -1]
    return x_, y_


def train():
    reset_random()

    x, y = get_data()
    ss = StandardScaler()
    x = ss.fit_transform(x)
    y = to_categorical(y, len(CLASSES))
    x_ex = np.expand_dims(x, axis=1)

    model_dir = 'model'
    # if os.path.isdir(model_dir):
    #     shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    acc_loss_csv_path = os.path.join(model_dir, "acc_loss.csv")
    model_path = os.path.join(model_dir, "model.h5")

    training_cb = TrainingCallback(acc_loss_csv_path, ACC_PLOT, LOSS_PLOT)
    checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=False,
    )

    model = get_ensemble_model(x_ex.shape[1:], x.shape[1])

    initial_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(acc_loss_csv_path):
        print("[INFO] Loading Pre-Trained Model :: {0}".format(model_path))
        model.load_weights(model_path)
        initial_epoch = len(pd.read_csv(acc_loss_csv_path))

    print("[INFO] Fitting Data")
    model.fit(
        [x_ex, x_ex, x],
        y,
        validation_data=([x_ex, x_ex, x], y),
        epochs=500,
        batch_size=8,
        verbose=1,
        initial_epoch=initial_epoch,
        callbacks=[training_cb, checkpoint],
    )

    model.load_weights(model_path)

    splits = {
        "80:20": 0.2,
        "70:30": 0.3,
    }

    for split in splits:
        print("[INFO] Evaluating Training|Testing ==> {0}".format(split))
        x, y = get_data()
        x = ss.transform(x)
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=splits[split], shuffle=True, random_state=1
        )
        train_x_ex = np.expand_dims(train_x, axis=1)
        train_prob = model.predict([train_x_ex, train_x_ex, train_x], verbose=False)
        train_pred = np.argmax(train_prob, axis=1)
        plot(
            train_y.ravel().astype(int),
            train_pred,
            train_prob,
            RESULTS_PLOT[split],
            "results/{0}/Train".format(split.replace(":", "-")),
            split.split(":")[0],
            split,
        )
        test_x_ex = np.expand_dims(test_x, axis=1)
        test_prob = model.predict([test_x_ex, test_x_ex, test_x], verbose=False)
        test_pred = np.argmax(test_prob, axis=1)
        plot(
            test_y.ravel().astype(int),
            test_pred,
            test_prob,
            RESULTS_PLOT[split],
            "results/{0}/Test".format(split.replace(":", "-")),
            split.split(":")[1],
            split,
        )


if __name__ == "__main__":
    train()

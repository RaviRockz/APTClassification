from keras import Input
from keras.layers import concatenate, LSTM, Bidirectional, GRU, Dense
from keras.models import Model, Sequential
from utils import CLASSES
from ChOA import ChimpOptimizer


def ensemble(models):
    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            layer.trainable = False
            layer._name = "ensemble_" + str(i + 1) + "_" + layer.name
    ensemble_visible = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]

    merge = concatenate(ensemble_outputs)
    hidden = Dense(16, activation="relu")(merge)
    output = Dense(len(CLASSES), activation="softmax")(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=ChimpOptimizer(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def lstm(shape):
    print("[INFO] Building LSTM Model")
    model = Sequential(name="LSTM")
    model.add(Input(shape=shape))
    model.add(LSTM(64, recurrent_dropout=0.2, return_sequences=False))
    return model


def bi_gru(shape):
    print("[INFO] Building Bidirectional-GRU Model")
    model = Sequential(name="BiGRU")
    model.add(Input(shape=shape))
    model.add(Bidirectional(GRU(64, recurrent_dropout=0.2, return_sequences=False)))
    return model


def encoder(in_dim):
    print("[INFO] Building Encoder")
    model = Sequential(name="encoder")
    model.add(Dense(64, input_dim=in_dim))
    model.add(Dense(32))
    return model


def decoder(out_dim):
    print("[INFO] Building Decoder")
    model = Sequential(name="decoder")
    model.add(Dense(64, input_dim=32))
    model.add(Dense(out_dim))
    return model


def autoencoder(in_dim):
    print("[INFO] Building AutoEncoder Model")
    enc = encoder(in_dim)
    dec = decoder(in_dim)
    model = Sequential([enc, dec], name="AE")
    return model


def get_ensemble_model(shape, in_dim):
    lstm_ = lstm(shape)
    bi_gru_ = bi_gru(shape)
    ae = autoencoder(in_dim)
    model = ensemble([lstm_, bi_gru_, ae])
    return model

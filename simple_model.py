import numpy as np
import matplotlib.pyplot as plt
import keras

from constants import START_DATE
from helpers import load_dataset, normalize_dataset, rolling_window, plot_predicted


def create_simple_dataset(x_data, y_data, window):
    x_data = x_data.reshape((x_data.shape[0], -1))
    x = rolling_window(x_data, window)
    y = y_data[window:]

    return x[:-1], y


def create_simple_model(num_inputs: int, num_outputs: int):
    inputs = keras.layers.Input(shape=(None, num_inputs), name="x_input")
    layer = keras.layers.GRU(64, return_sequences=True)(inputs)
    layer = keras.layers.GRU(32)(layer)
    outputs = keras.layers.Dense(num_outputs, name="y_output")(layer)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    keras.utils.plot_model(
        model,
        to_file="simple_model.png",
        show_shapes=True,
        show_layer_names=False,
        rankdir="LR",
        dpi=300,
    )

    return model


def plot_training_history_with_validation(history, with_validation: bool = False):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"])
    axes[0].set_ylabel("MSE")
    axes[0].set_xlabel('Епоха')

    axes[1].plot(history.history[f"mae"])
    axes[1].set_ylabel("MAE")
    axes[1].set_xlabel('Епоха')

    if with_validation:
        axes[0].plot(history.history["val_loss"])
        axes[1].plot(history.history["val_mae"])

    axes[0].legend(["Навчання", "Тест"])
    axes[1].legend(["Навчання", "Тест"])
    plt.show()


def predict_values(model: keras.Model, x_data: np.ndarray, days: int = 28):
    x_pred = np.copy(x_data)

    for _ in range(days):
        x_pred_next = model.predict(x_pred.reshape(1, *x_pred.shape))
        x_pred = np.vstack([x_pred, x_pred_next])

    return x_pred


def run(x_key, y_key, start_from: int = 0, window: int = 14, validation_split=.0):
    x_data = load_dataset(x_key)[start_from:]
    y_data = load_dataset(y_key)[start_from:]
    x_data, x_mean, x_std = normalize_dataset(x_data)
    y_data, y_mean, y_std = normalize_dataset(y_data)

    x, y = create_simple_dataset(x_data, y_data, window)

    checkpoint = keras.callbacks.ModelCheckpoint(
        f"{x_key}_{y_key}_{window}_simple.hdf5",
        monitor="loss",
        verbose=True,
        save_best_only=True,
    )
    callback = keras.callbacks.EarlyStopping(patience=20, monitor="loss", verbose=True, restore_best_weights=True)

    model = create_simple_model(x.shape[-1], y.shape[-1])

    history = model.fit(
        x=x,
        y=y,
        validation_split=validation_split,
        epochs=100,
        batch_size=x.shape[0],
        callbacks=[callback],
    )

    plot_training_history_with_validation(history, with_validation=bool(validation_split))

    if x.shape[-1] == y.shape[-1]:

        y_pred = predict_values(model, x_data, 28)
        plot_predicted(y_pred * y_std + y_mean, START_DATE.shift(days=start_from), f"{x_key}_{y_key}_{window}_simple.png")


if __name__ == '__main__':
    x_key = "oblast"
    y_key = "ukraine"
    start_from = 41
    window = 14

    run(x_key, y_key, start_from, window, 0.2)

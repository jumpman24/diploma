import keras
import numpy as np
import matplotlib.pyplot as plt
from constants import START_DATE
from helpers import (
    load_dataset,
    normalize_dataset,
    plot_predicted,
    rolling_window,
)


def create_dataset_multi_step(x_data, y_data, window):
    x_data = x_data.reshape((x_data.shape[0], -1))
    x = rolling_window(x_data, window)
    y1 = x_data[window:]
    y2 = y_data[window:]

    return x[:-1], y1, y2


def create_multi_model(num_inputs, num_outputs, optimizer, name):
    inputs = keras.layers.Input(shape=(None, num_inputs), name="x_input")
    layer = keras.layers.GRU(64, name="gru_1", return_sequences=True, return_state=True, recurrent_dropout=.2)(inputs)
    layer = keras.layers.GRU(64, name="gru_2", return_sequences=True, recurrent_dropout=.2)(layer)
    layer = keras.layers.GRU(128, name="gru_3", recurrent_dropout=.2)(layer)
    layer = keras.layers.Dense(256, name="dense_1")(layer)

    output1 = keras.layers.Dense(num_inputs, name="x_output")(layer)
    output2 = keras.layers.Dense(num_outputs, name="y_output")(layer)

    model = keras.Model(inputs=inputs, outputs=[output1, output2])
    model.compile(optimizer, loss="mse", metrics="mae")
    model.summary()
    keras.utils.plot_model(
        model,
        to_file=f"model_{name}.png",
        show_shapes=True,
        show_layer_names=False,
        rankdir="LR",
        dpi=300,
    )
    return model


def plot_training_history_with_validation(history, name):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["x_output_loss"])
    axes[0].plot(history.history["y_output_loss"])
    axes[0].legend(["x_output", "y_output"])
    axes[0].set_xlabel('Епохи')
    axes[0].set_ylabel("MSE")
    axes[0].axvline(np.argmin(history.history["loss"]), linestyle="dashed")

    axes[1].plot(history.history[f"x_output_mae"])
    axes[1].plot(history.history[f"y_output_mae"])
    axes[1].legend(["x_output", "y_output"])
    axes[1].set_xlabel('Епохи')
    axes[1].set_ylabel("MAE")

    plt.savefig(f"loss_{name}.png")
    plt.show()


def predict_multi_values(model, x, y, predictions):
    x, y = np.copy(x), np.copy(y)
    for _ in range(predictions):
        x_pred, y_pred = model.predict(x.reshape(1, *x.shape))
        x, y = np.vstack([x, x_pred]), np.vstack([y, y_pred])

    return x, y


def run(x_key, y_key, start_from: int = 0, window: int = 14, epochs: int = 100, optimizer="adam"):
    x_data, x_mean, x_std = normalize_dataset(load_dataset(x_key)[start_from:])
    y_data, y_mean, y_std = normalize_dataset(load_dataset(y_key)[start_from:])

    x, y1, y2 = create_dataset_multi_step(x_data, y_data, window)

    if isinstance(optimizer, keras.optimizers.Optimizer):
        optimizer_name = optimizer.get_config()["name"]
    else:
        optimizer_name = optimizer

    filename = f"{x_key}_{y_key}_{window}_{optimizer_name}"
    checkpoint = keras.callbacks.ModelCheckpoint(
        f"weights_{filename}.hdf5",
        monitor="loss",
        verbose=True,
        save_best_only=True,
    )
    callback = keras.callbacks.EarlyStopping(patience=20, monitor="loss", verbose=True, restore_best_weights=True)

    model = create_multi_model(x.shape[-1], y2.shape[-1], optimizer, name=filename)
    history = model.fit(x=x, y=[y1, y2], epochs=epochs, batch_size=x.shape[0], callbacks=[callback, checkpoint])
    plot_training_history_with_validation(history, filename)

    _, y_pred = predict_multi_values(model, x_data, y_data, 28)

    plot_predicted(y_pred * y_std + y_mean, START_DATE.shift(days=start_from), f"graph_{filename}.png")


if __name__ == '__main__':
    x_key = "combined"
    y_key = "ukraine"
    start_from = 41
    window = 14
    epochs = 500

    run(x_key, y_key, start_from, window, epochs, "rmsprop")

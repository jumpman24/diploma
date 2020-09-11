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


def create_multi_model(num_inputs, num_outputs, optimizer):
    inputs = keras.layers.Input(shape=(None, num_inputs), name="x_input")
    layer = keras.layers.GRU(128, name="gru_1", return_sequences=True, return_state=True, recurrent_dropout=.1)(inputs)
    layer = keras.layers.GRU(128, name="gru_2", return_sequences=True, recurrent_dropout=.1)(layer)
    layer = keras.layers.GRU(256, name="gru_3", recurrent_dropout=.1)(layer)
    layer = keras.layers.Dense(256, name="dense_1")(layer)

    output1 = keras.layers.Dense(num_inputs, name="x_output")(layer)
    output2 = keras.layers.Dense(num_outputs, name="y_output")(layer)

    model = keras.Model(inputs=inputs, outputs=[output1, output2])
    model.compile(optimizer, loss="mse", metrics="mae")
    model.summary()
    keras.utils.plot_model(
        model,
        to_file="multi_model.png",
        show_shapes=True,
        show_layer_names=False,
        rankdir="LR",
        dpi=300,
    )
    return model


def plot_training_history_with_validation(history):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["x_output_loss"])
    axes[0].plot(history.history["y_output_loss"])
    axes[0].legend(["x_output", "y_output"])
    axes[0].set_xlabel('Епохи')
    axes[0].set_ylabel("MSE")

    axes[1].plot(history.history[f"x_output_mae"])
    axes[1].plot(history.history[f"y_output_mae"])
    axes[1].legend(["x_output", "y_output"])
    axes[1].set_xlabel('Епохи')
    axes[1].set_ylabel("MAE")

    plt.show()


def predict_multi_values(model, x, y, predictions):
    x, y = np.copy(x), np.copy(y)
    for _ in range(predictions):
        x_pred, y_pred = model.predict(x.reshape(1, *x.shape))
        x, y = np.vstack([x, x_pred]), np.vstack([y, y_pred])

    return x, y


def run(x_key, y_key, start_from: int = 0, window: int = 14, optimizer="adam"):
    x_data = load_dataset(x_key)[start_from:]
    y_data = load_dataset(y_key)[start_from:]
    x_data, x_mean, x_std = normalize_dataset(x_data)
    y_data, y_mean, y_std = normalize_dataset(y_data)

    x, y1, y2 = create_dataset_multi_step(x_data, y_data, window)

    if isinstance(optimizer, keras.optimizers.Optimizer):
        optimizer_name = optimizer.get_config()["name"]
    else:
        optimizer_name = optimizer

    checkpoint = keras.callbacks.ModelCheckpoint(
        f"{x_key}_{y_key}_{window}_{optimizer_name}.hdf5",
        monitor="loss",
        verbose=True,
        save_best_only=True,
    )
    callback = keras.callbacks.EarlyStopping(patience=50, monitor="loss", verbose=True, restore_best_weights=True)

    model = create_multi_model(x.shape[-1], y2.shape[-1], optimizer)
    history = model.fit(x=x, y=[y1, y2], epochs=200, batch_size=x.shape[0], callbacks=[callback, checkpoint])
    plot_training_history_with_validation(history)

    _, y_pred = predict_multi_values(model, x_data, y_data, 28)

    plot_predicted(y_pred * y_std + y_mean, START_DATE.shift(days=start_from), f"{x_key}_{y_key}_{window}_{optimizer_name}.png")


if __name__ == '__main__':
    x_key = "combined"
    y_key = "ukraine"
    start_from = 41
    window = 14
    optimizer = keras.optimizers.RMSprop(rho=0.8)

    run(x_key, y_key, start_from, window, optimizer)

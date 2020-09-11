from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

from nn import create_deep_model, create_multi_model
from helpers import (
    load_dataset,
    normalize_dataset,
    plot_training_history_with_validation,
    predict_values,
    plot_predicted,
    create_train_test,
    create_dataset_multi_step,
    predict_multi_values,
)

start_from = 41
x_window = 14
y_window = 1
batch_size = None
# x_data = load_dataset("ukraine")[start_from:]
# y_data = load_dataset("ukraine")[start_from:]
# x_data, x_mean, x_std = normalize_dataset(x_data)
# y_data, y_mean, y_std = normalize_dataset(y_data)
#
# # train, test = create_train_test(
# #     x_data, y_data,
# #     x_window=x_window,
# #     y_window=y_window,
# #     batch_size=batch_size,
# # )
#
# x, y1, y2 = create_dataset_multi_step(x_data, y_data, x_window)
#
# checkpoint = ModelCheckpoint("best_ukraine_model.hdf5", monitor="loss", verbose=True, save_best_only=True)
# callback = EarlyStopping(patience=50, monitor="loss", verbose=True, restore_best_weights=True)
#
# model = create_multi_model(3, 3, keras.optimizers.RMSprop(rho=0.75))
#
# history = model.fit(x=x, y=[y1, y2], epochs=200, batch_size=x.shape[0], callbacks=[callback, checkpoint])
#
# plot_training_history_with_validation(history)
#
#
# ukraine_data = load_dataset("ukraine")
# ukraine_data, mean, std = normalize_dataset(ukraine_data)
#
# predictions = predict_values(model, ukraine_data, 28)
# plot_predicted(predictions*std+mean, 28)


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

    checkpoint = ModelCheckpoint(
        f"{x_key}_{y_key}_{window}_{optimizer_name}.hdf5",
        monitor="loss",
        verbose=True,
        save_best_only=True,
    )
    callback = EarlyStopping(patience=100, monitor="loss", verbose=True, restore_best_weights=True)

    model = create_multi_model(x.shape[-1], y2.shape[-1], optimizer)
    history = model.fit(x=x, y=[y1, y2], epochs=200, batch_size=x.shape[0], callbacks=[callback, checkpoint])
    plot_training_history_with_validation(history)

    x_pred, y_pred = predict_multi_values(model, x_data, y_data, 28)
    plot_predicted(y_pred[-128:] * y_std + y_mean, 28)


if __name__ == '__main__':
    x_key = "ukraine"
    y_key = "ukraine"
    start_from = 41
    window = 28
    optimizer = keras.optimizers.RMSprop(rho=0.8)

    run(x_key, y_key, start_from, window, optimizer)

import h5py
import numpy as np
import keras
import matplotlib.pyplot as plt
import datetime


def load_dataset(key):
    with h5py.File("dataset.h5", mode="r") as file:
        return np.array(file[key])


def normalize_dataset(dataset, axis=0):
    mean = dataset.mean(axis=axis)
    std = dataset.std(axis=axis)
    normalized = (dataset - mean) / std
    return normalized.reshape(normalized.shape[0], -1), mean, std


def rolling_window(a, window):
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0], ) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def create_dataset_multi_step(x_data, y_data, window):
    x_data = x_data.reshape((x_data.shape[0], -1))
    x = rolling_window(x_data, window)
    y1 = x_data[window:]
    y2 = y_data[window:]

    return x[:-1], y1, y2


def create_dataset_multi_step(x_data, y_data, window):
    x_data = x_data.reshape((x_data.shape[0], -1))
    x = rolling_window(x_data, window)
    y1 = x_data[window:]
    y2 = y_data[window:]

    return x[:-1], y1, y2


def create_targets(data, window):
    targets = keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=window,
        batch_size=data.shape[0]
    )

    for dataset in targets:
        return np.array(dataset).reshape(dataset.shape[0], -1)


def create_train_test(x_data, y_data, *,
                      x_window: int = 14,
                      y_window: int = 1,
                      train_split: float = 0.8,
                      batch_size=None):

    if y_window > 1:
        y_data = create_targets(y_data, y_window)

    split_idx = int((x_data.shape[0] - x_window) * train_split)
    train_x, train_y, = x_data[:split_idx], y_data[:split_idx]
    test_x, test_y = x_data[split_idx:], y_data[split_idx:]

    batch_size = batch_size or x_data.shape[0]
    train_dataset = keras.preprocessing.timeseries_dataset_from_array(
        data=train_x[:-(x_window+y_window-1)],
        targets=train_y[x_window:],
        sequence_length=x_window,
        batch_size=batch_size,
    )
    test_dataset = keras.preprocessing.timeseries_dataset_from_array(
        data=test_x[:-(x_window+y_window-1)],
        targets=test_y[x_window:],
        sequence_length=x_window,
        batch_size=batch_size,
    )

    return train_dataset, test_dataset


def plot_training_history_with_validation(history, key: str = "mae", axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"])
    axes[0].plot(history.history["dense_1_loss"])
    axes[0].plot(history.history["dense_2_loss"])
    axes[0].set_ylabel("LOSS")
    axes[0].set_xlabel('Epoch')

    axes[1].plot(history.history[f"dense_1_{key}"])
    axes[1].plot(history.history[f"dense_2_{key}"])
    axes[1].set_ylabel(key.upper())
    axes[1].set_xlabel('Epoch')
    plt.show()


def predict_values(model, dataset, predictions, lags=0):
    result = np.copy(dataset)

    for _ in range(predictions):
        predicted = model.predict(result[-lags:].reshape(1, *result[-lags:].shape))
        result = np.vstack([result, predicted])

    return result


def predict_multi_values(model, x, y, predictions):
    x = np.copy(x)
    y = np.copy(y)
    for _ in range(predictions):
        y1, y2 = model.predict(x.reshape(1, *x.shape))
        x = np.vstack([x, y1])
        y = np.vstack([y, y2])

    return x, y


def plot_predicted(values, pred_num: int):
    labels = ["Хворіють", "Померли", "Одужали"]
    colors = ["r", "b", "g"]

    today = datetime.date.today()
    x = np.array([today - datetime.timedelta(i-pred_num) for i in reversed(range(0, values.shape[0]))])

    cum_values = np.cumsum(values, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, color, arr, cum_arr in zip(labels, colors, values.T, cum_values.T):
        mask_pred = np.ma.masked_where(x <= today, arr)
        mask_real = np.ma.masked_where(x > today, arr)

        mask_cum_pred = np.ma.masked_where(x <= today, cum_arr)
        mask_cum_real = np.ma.masked_where(x > today, cum_arr)

        axes[0].plot(x, mask_real, f"{color}", x, mask_pred, f"{color}--")
        axes[1].plot(x, mask_cum_real, f"{color}", x, mask_cum_pred, f"{color}--")

    axes[0].legend(["Хворіють", None, "Померли", None, "Одужали", None])
    axes[1].legend(["Хворіють", None, "Померли", None, "Одужали", None])
    fig.autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    data = load_dataset("ukraine")
    targets = create_targets(data, 7)
    print(targets.shape)

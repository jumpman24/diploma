import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import arrow


def load_dataset(key):
    with h5py.File("dataset.h5", mode="r") as file:
        return np.array(file[key])


def normalize_dataset(dataset):
    axis = tuple(i for i in range(dataset.ndim-1))
    mean = dataset.mean(axis=axis)
    std = dataset.std(axis=axis)
    normalized = (dataset - mean) / std
    return normalized.reshape(normalized.shape[0], -1), mean, std


def rolling_window(a, window):
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0], ) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plot_predicted(values, first_date, save_to: str):
    labels = ["Хворіють", "Померли", "Одужали"]
    colors = ["r", "b", "g"]

    today = arrow.utcnow()
    x_ticks = np.array([first_date.shift(days=i).datetime for i in range(len(values))])
    cum_values = np.cumsum(values, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x_lines = []
    x_cum_lines = []
    for label, color, arr, cum_arr in zip(labels, colors, values.T, cum_values.T):
        mask_pred = np.ma.masked_where(x_ticks <= today, arr)
        mask_real = np.ma.masked_where(x_ticks > today, arr)

        mask_cum_pred = np.ma.masked_where(x_ticks <= today, cum_arr)
        mask_cum_real = np.ma.masked_where(x_ticks > today, cum_arr)

        x_real, x_pred = axes[0].plot(x_ticks, mask_real, f"{color}", x_ticks, mask_pred, f"{color}--")
        x_real_cum, x_pred_cum = axes[1].plot(x_ticks, mask_cum_real, f"{color}", x_ticks, mask_cum_pred, f"{color}--")
        x_lines.append((x_real, x_pred))
        x_cum_lines.append((x_real_cum, x_pred_cum))

    axes[0].legend(x_lines, labels, handler_map={tuple: HandlerTuple(2)})
    axes[1].legend(x_cum_lines, labels, handler_map={tuple: HandlerTuple(2)})

    fig.autofmt_xdate()
    plt.savefig(save_to)
    plt.show()

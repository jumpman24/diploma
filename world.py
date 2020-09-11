from keras.callbacks import EarlyStopping, ModelCheckpoint

from nn import create_deep_model
from helpers import load_dataset, normalize_dataset, create_train_test_datasets, plot_training_history_with_validation


world_data, world_mean, world_std = normalize_dataset(load_dataset("world"), (0, 1))
ukraine_data, ukraine_mean, ukraine_std = normalize_dataset(load_dataset("ukraine"))

window = 14
train, test = create_train_test_datasets(world_data, ukraine_data, x_window=window)


checkpoint = ModelCheckpoint("best_world_model.hdf5", verbose=1, save_best_only=True, mode='auto', period=1)
callback = EarlyStopping(patience=10, restore_best_weights=True)
model = create_deep_model(world_data.shape[1], 3)
history = model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint])

plot_training_history_with_validation(history)

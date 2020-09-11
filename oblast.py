from keras.callbacks import EarlyStopping, ModelCheckpoint

from nn import create_deep_model, create_simple_model
from helpers import load_dataset, normalize_dataset, create_train_test_datasets, plot_training_history_with_validation, predict_values, plot_predicted


oblast_data, _, _ = normalize_dataset(load_dataset("oblast"), (0, 1))
ukraine_data, mean, std = normalize_dataset(load_dataset("ukraine"))

window = 14
train, test = create_train_test_datasets(oblast_data, oblast_data, x_window=window)


checkpoint = ModelCheckpoint("best_oblast_model.hdf5", verbose=True, save_best_only=True, period=1)
callback = EarlyStopping(patience=10, verbose=True, restore_best_weights=True)
model = create_simple_model(75, 3)
history = model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint, callback])

model.load_weights("best_oblast_model.hdf5")
plot_training_history_with_validation(history)

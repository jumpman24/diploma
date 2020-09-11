import keras


def create_simple_model(num_inputs: int, num_outputs: int):
    inputs = keras.layers.Input(shape=(None, num_inputs))
    layer = keras.layers.GRU(64, return_sequences=True)(inputs)
    layer = keras.layers.GRU(32)(layer)
    outputs = keras.layers.Dense(num_outputs)(layer)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    return model


def create_deep_model(num_inputs, num_outputs, optimizer="adam", loss="mse", metrics=("mae",)):
    inputs = keras.layers.Input(shape=(None, num_inputs))
    layer = keras.layers.GRU(128, return_sequences=True, return_state=True, recurrent_dropout=.1)(inputs)
    layer = keras.layers.GRU(128, return_sequences=True, recurrent_dropout=.1)(layer)
    layer = keras.layers.GRU(256, recurrent_dropout=.1)(layer)
    layer = keras.layers.Dense(256)(layer)
    outputs = keras.layers.Dense(num_outputs)(layer)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer, loss=loss, metrics=metrics)
    model.summary()
    keras.utils.plot_model(
        model,
        show_shapes=True,
        show_layer_names=False,
        rankdir="LR",
        dpi=300,
    )
    return model


def create_multi_model(num_inputs, num_outputs, optimizer="adam", loss="mse", metrics=("mae",)):
    inputs = keras.layers.Input(shape=(None, num_inputs), name="x_input")
    layer = keras.layers.GRU(128, name="gru_1", return_sequences=True, return_state=True, recurrent_dropout=.2)(inputs)
    layer = keras.layers.GRU(128, name="gru_2", return_sequences=True, recurrent_dropout=.2)(layer)
    layer = keras.layers.GRU(256, name="gru_3", recurrent_dropout=.2)(layer)
    layer = keras.layers.Dense(256, name="dense_1")(layer)
    # layer = keras.layers.Dense(256, name="dense_2")(layer)

    output1 = keras.layers.Dense(num_inputs, name="x_output")(layer)
    output2 = keras.layers.Dense(num_outputs, name="y_output")(layer)

    model = keras.Model(inputs=inputs, outputs=[output1, output2])
    model.compile(optimizer, loss=loss, metrics=metrics)
    model.summary()
    keras.utils.plot_model(
        model,
        show_shapes=True,
        show_layer_names=False,
        rankdir="LR",
        dpi=300,
    )
    return model


if __name__ == '__main__':
    create_deep_model(75, 3)

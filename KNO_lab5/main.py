import tensorflow as tf
from tensorflow.keras import layers, Model
import keras_tuner as kt
import numpy as np


# Customowa klasa modelu
class CustomModel(Model):
    def __init__(self, num_hidden_units, dropout_rate):
        super(CustomModel, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(num_hidden_units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(10, activation='softmax')  # Przykład dla klasyfikacji 10-klasowej

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)


def build_model(hp):
    # Hiperparametry do strojenia
    num_hidden_units = hp.Int('num_hidden_units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = CustomModel(num_hidden_units, dropout_rate)
    model.build(input_shape=(None, 28, 28))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    # Przygotowanie tunera
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,  # Maksymalna liczba epok dla każdej rundy
        factor=3,  # Czynnik redukcji liczby modeli w kolejnych rundach
        directory='my_tuner',
        project_name='custom_model_tuning'
    )

    # Przykładowe dane do treningu (MNIST jako placeholder)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Callback do wczesnego zatrzymania
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Uruchomienie strojenia
    tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])

    # Wybranie najlepszego modelu
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Optymalne hiperparametry: {best_hps.values}")

    # Trenowanie modelu z najlepszymi hiperparametrami
    final_model = tuner.hypermodel.build(best_hps)
    history = final_model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[stop_early])

    # Ewaluacja modelu
    loss, accuracy = final_model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")


if __name__ == "__main__":
    main()

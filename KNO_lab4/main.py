import datetime
import pandas as pd
from keras import Input
from keras.layers import Dense, Dropout
from sklearn.utils import shuffle
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import itertools


dataset = pd.read_csv("wine.data", header=None)
dataset.columns = [
    "class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols",
    "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue",
    "0D280_0D315_of_diluted_wines", "Proline"
]
dataset = shuffle(dataset, random_state=1337)

# Podziel dane na zbiory treningowy, walidacyjny i testowy
train_dataframe = dataset.sample(frac=0.6, random_state=1337)
remaining_dataframe = dataset.drop(train_dataframe.index)
val_dataframe = remaining_dataframe.sample(frac=0.5, random_state=1337)
test_dataframe = remaining_dataframe.drop(val_dataframe.index)


# Funkcja konwertująca DataFrame do formatu tf.data.Dataset
def dataframe_to_dataset(dataframe, loss_function):
    dataframe = dataframe.copy()
    labels = dataframe.pop("class")
    if loss_function in ["categorical_crossentropy", "categorical_focal_crossentropy"]:
        # One-hot encoding etykiet dla categorical funkcji strat
        labels = keras.utils.to_categorical(labels.values - 1, num_classes=3)
    else:
        # Etykiety w formie liczb całkowitych dla sparse_categorical_crossentropy
        labels = labels.values - 1
    ds = tf.data.Dataset.from_tensor_slices((dataframe.values, labels))
    ds = ds.shuffle(buffer_size=len(dataframe)).batch(32)
    return ds


# Funkcja tworząca model sieci neuronowej
def create_model(input_shape, layer_config, dropout_rate=0.0):
    model = keras.Sequential()
    model.add(Input(shape=(input_shape,)))
    for units, activation in layer_config:
        model.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))  # Warstwa wyjściowa
    return model


# Definicje układów warstw dla obu modeli
layer_config_1 = [
    (13 * 2, 'relu'),
    (13, 'relu')
]

layer_config_2 = [
    (13 * 2, 'relu'),
    (13, 'selu'),
    (13, 'elu')
]

# Parametry do optymalizacji
learning_rates = [0.001, 0.01]
dropout_rates = [0.0, 0.2]
loss_functions = ["categorical_crossentropy", "categorical_focal_crossentropy"]

# Modelowanie
input_shape = train_dataframe.shape[1] - 1
results = []

# Generowanie wszystkich kombinacji parametrów
combinations = list(itertools.product(learning_rates, dropout_rates, loss_functions))


# Funkcja do trenowania i ewaluacji modelu
def train_and_evaluate_model(model, train_ds, val_ds, test_ds, lr, dropout_rate, model_name, loss):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])
    history = model.fit(train_ds, epochs=100, validation_data=val_ds, verbose=0)

    val_acc = max(history.history['val_accuracy'])
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Strata treningowa')
    plt.plot(history.history['val_loss'], label='Strata walidacyjna')
    plt.title(f'Strata dla {model_name} (LR: {lr}, Dropout: {dropout_rate})')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Dokładność treningowa')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
    plt.title(f'Dokładność dla {model_name} (LR: {lr}, Dropout: {dropout_rate})')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.show()

    return {
        'model': model_name,
        'learning_rate': lr,
        'dropout_rate': dropout_rate,
        'validation_accuracy': val_acc,
        'loss_function': loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }


# Trenowanie modeli z różnymi kombinacjami parametrów
for lr, dropout_rate, loss_function in combinations:
    train_ds = dataframe_to_dataset(train_dataframe, loss_function)
    val_ds = dataframe_to_dataset(val_dataframe, loss_function)
    test_ds = dataframe_to_dataset(test_dataframe, loss_function)

    model_1 = create_model(input_shape, layer_config_1, dropout_rate=dropout_rate)
    result_1 = train_and_evaluate_model(model_1, train_ds, val_ds, test_ds, lr, dropout_rate, 'Model 1', loss_function)
    results.append(result_1)

    model_2 = create_model(input_shape, layer_config_2, dropout_rate=dropout_rate)
    result_2 = train_and_evaluate_model(model_2, train_ds, val_ds, test_ds, lr, dropout_rate, 'Model 2', loss_function)
    results.append(result_2)

results_df = pd.DataFrame(results)
print(results_df)

best_result = results_df.loc[results_df['validation_accuracy'].idxmax()]
print("\nNajlepszy model:")
print(best_result)

best_model = create_model(input_shape, layer_config_1 if best_result['model'] == 'Model 1' else layer_config_2,
                          dropout_rate=best_result['dropout_rate'])
best_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_result['learning_rate']),
                   loss=best_result['loss_function'], metrics=["accuracy"])

train_ds = dataframe_to_dataset(train_dataframe, best_result['loss_function'])
val_ds = dataframe_to_dataset(val_dataframe, best_result['loss_function'])
best_history = best_model.fit(train_ds, validation_data=val_ds, epochs=100, verbose=0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(best_history.history['loss'], label='Strata treningowa')
plt.plot(best_history.history['val_loss'], label='Strata walidacyjna')
plt.title('Strata dla najlepszego modelu')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_history.history['accuracy'], label='Dokładność treningowa')
plt.plot(best_history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Dokładność dla najlepszego modelu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.show()

final_test_loss, final_test_acc = best_model.evaluate(test_ds, verbose=0)
print(f"\nWynik najlepszego modelu na zbiorze testowym - Loss: {final_test_loss}, Accuracy: {final_test_acc}")

print("Trenowanie i ewaluacja modeli zakończone.")
tf.keras.backend.clear_session()

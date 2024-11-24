import datetime

import pandas as pd
from keras import Input
from keras.layers import LayerNormalization, Activation, Dense
from sklearn.utils import shuffle
import keras
import tensorflow as tf

dataset = pd.read_csv("wine.data")
dataset = shuffle(dataset)
dataset.columns = [
    "class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols",
    "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue",
    "0D280_0D315_of_diluted_wines", "Proline"
]
val_dataframe = dataset.sample(frac=0.2, random_state=1337)
train_dataframe = dataset.drop(val_dataframe.index)


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("class")
    ds = tf.data.Dataset.from_tensor_slices((dataframe.values, keras.utils.to_categorical(labels.values - 1,
                                                                                          num_classes=3)))  # Convert to (features, labels) tuple
    ds = ds.shuffle(buffer_size=len(dataframe)).batch(32)  # Shuffle and batch the dataset
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

input_shape = 0
for x, y in train_ds.take(1):
    input_shape = x.shape[1]

model_1 = keras.Sequential([
    Input(shape=(input_shape,)),  # Warstwa wejściowa
    Dense(13 * 2, activation='relu'),  # Pierwsza warstwa gęsta
    Dense(13, activation='relu'),  # Druga warstwa gęsta
    Dense(3, activation='softmax')  # Warstwa wyjściowa z softmax dla 3 klas
])

model_2 = keras.Sequential([
    Input(shape=(input_shape,)),  # Warstwa wejściowa
    Dense(13 * 2, activation='relu'),  # Pierwsza warstwa gęsta
    Dense(13, activation='selu'),  # Druga warstwa gęsta
    Dense(13, activation='elu'),  # Trzecia warstwa gęsta
    Dense(3, activation='softmax')  # Warstwa wyjściowa z softmax dla 3 klas
])

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_1.fit(train_ds, epochs=200, validation_data=val_ds,
            callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir + "1", histogram_freq=1)])

model_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_2.fit(train_ds, epochs=200, validation_data=val_ds,
            callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir + "2", histogram_freq=1)])

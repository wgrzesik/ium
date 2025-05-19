import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

train_data = pd.read_csv("data/AB_NYC_2019_train.csv")
dev_data = pd.read_csv("data/AB_NYC_2019_dev.csv")
test_data = pd.read_csv("data/AB_NYC_2019_test.csv")

X_train = train_data.drop('price', axis=1)
y_train = train_data['price']

X_dev = dev_data.drop('price', axis=1)
y_dev = dev_data['price']

X_test = test_data.drop('price', axis=1)
y_test = test_data['price']

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error', optimizer=optimizer)

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, batch_size=batch_size)

model.save("data/airbnb_price_model.h5")
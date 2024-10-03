import tensorflow as tf
import numpy as np

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y_train = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

model.fit(x_train, y_train, epochs=500)

while True:
    try:
        number = float(input(": "))
        predicted = model.predict(np.array([number], dtype=float))
        print(predicted)
    except ValueError as e:
        print(e)
        break

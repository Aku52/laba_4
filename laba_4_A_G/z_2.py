import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal, RandomUniform, TruncatedNormal


# Модель с нормальным распределением
model_1 = Sequential([
    Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), input_shape=(100,)),
    Dense(32, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05)),
    Dense(10, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))
])

# Модель с равномерным распределением
model_2 = Sequential([
    Dense(64, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05), input_shape=(100,)),
    Dense(32, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05)),
    Dense(10, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05))
])

# Модель с усечённым нормальным распределением
model_3 = Sequential([
    Dense(64, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05), input_shape=(100,)),
    Dense(32, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05)),
    Dense(10, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05))
])

# Теперь веса будут доступны
weights_1, biases_1 = model_1.layers[0].get_weights()
weights_2, biases_2 = model_2.layers[0].get_weights()
weights_3, biases_3 = model_3.layers[0].get_weights()



# Строим гистограмму весов
plt.hist(weights_1.flatten(), bins=30, color='blue', edgecolor='black')
plt.hist(weights_3.flatten(), bins=30, color='red', edgecolor='black')
plt.hist(weights_2.flatten(), bins=30, color='green', edgecolor='black')


# Подписи
plt.title("Распределение весов")
plt.xlabel("Значение весов")
plt.ylabel("Количество")

plt.show()

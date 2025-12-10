import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal, RandomUniform, TruncatedNormal
from keras.regularizers import l2
from scipy import stats
from keras.datasets import mnist

# Загружаем данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:200].reshape((200, 28*28)).astype("float32") / 255
y_train = y_train[:200]

# Функция для обучения и получения весов
def build_and_train(initializer, reg_strength=0.01):
    model = Sequential([
        Dense(64, kernel_initializer=initializer, 
              kernel_regularizer=l2(reg_strength), input_shape=(784,)),
        Dense(32, kernel_initializer=initializer, 
              kernel_regularizer=l2(reg_strength)),
        Dense(10, kernel_initializer=initializer, 
              kernel_regularizer=l2(reg_strength))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    # Берём веса первого слоя
    weights, biases = model.layers[0].get_weights()
    return weights.flatten()

# Три модели с разными инициализациями + L2-регуляризация
w1 = build_and_train(RandomNormal(mean=0.0, stddev=0.05))
w2 = build_and_train(RandomUniform(minval=-0.05, maxval=0.05))
w3 = build_and_train(TruncatedNormal(mean=0.0, stddev=0.05))

# Гистограммы
plt.hist(w1, bins=30, color='blue', label='RandomNormal')
plt.hist(w3, bins=30, color='red', label='TruncatedNormal')
plt.hist(w2, bins=30, color='green',  label='RandomUniform')

plt.title("Распределение весов первого слоя (с L2)")
plt.xlabel("Значение весов")
plt.ylabel("Количество")
plt.legend()
plt.show()

# Описательные статистики
print("RandomNormal: mean =", np.mean(w1), "std =", np.std(w1))
print("RandomUniform: mean =", np.mean(w2), "std =", np.std(w2))
print("TruncatedNormal: mean =", np.mean(w3), "std =", np.std(w3))

# ANOVA
f_stat, p_value = stats.f_oneway(w1, w2, w3)
print("\nANOVA F-статистика:", f_stat)
print("ANOVA p-значение:", p_value)

# Kruskal-Wallis
h_stat, p_kw = stats.kruskal(w1, w2, w3)
print("\nKruskal-Wallis H-статистика:", h_stat)
print("Kruskal-Wallis p-значение:", p_kw)

if p_value < 0.05 or p_kw < 0.05:
    print("\nВывод: различия статистически значимы")
else:
    print("\nВывод: различия незначимы")

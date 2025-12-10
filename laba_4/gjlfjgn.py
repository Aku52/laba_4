import numpy as np
from keras import models, layers, backend as K
from keras.datasets import mnist
from keras.callbacks import Callback

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:200].reshape((200, 28*28)).astype("float32") / 255
y_train = y_train[:200]

# Создание модели
model = models.Sequential([
    layers.Dense(1024, activation='relu', input_shape=(28*28,)),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(1024, activation='relu'),  
    layers.Dense(10, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# Callback для мониторинга градиентов
class SimpleGradientMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # собираем все обучаемые веса
        weights = sum([layer.trainable_weights for layer in self.model.layers if layer.trainable_weights], [])
        if not weights:
            return
        
        # функция для вычисления градиентов
        grads_fn = K.function([self.model.input, self.model.targets, K.learning_phase()],
                              K.gradients(self.model.total_loss, weights))
        
        # берём небольшой батч
        batch_x, batch_y = x_train[:32], y_train[:32]
        grads_values = grads_fn([batch_x, batch_y, 0])
        
        # считаем нормы
        norms = [np.linalg.norm(g) for g in grads_values if g is not None]
        print(f"Эпоха {epoch+1}: средняя = {np.mean(norms):.6f}, макс = {np.max(norms):.6f}")

# Обучение с мониторингом
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    verbose=1,
    callbacks=[SimpleGradientMonitor()]
)

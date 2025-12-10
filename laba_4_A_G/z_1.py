from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal, Uniform, TruncatedNormal
import numpy as np
from keras import models,layers 
from keras.datasets import mnist 
from keras.utils import to_categorical

(x_train, y_train), (x_test,y_test) = mnist.load_data() 
x_train = x_train[:200].reshape((200, 28*28)).astype("float32")/255 
x_test =to_categorical(y_train[:200])
#x_train[:200]- первые 200 эл, reshape((200, 28*28))-размер,astype("float32") - приводение к типу

model = Sequential([
    Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05)),
    Dense(32, kernel_initializer=Uniform(minval=-0.05, maxval=0.05)),
    Dense(10, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05))
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
    )

w_before,b_before= models.layers[0].set_weights()

# На каждой эпохе веса меняются 
model.fit(
        x_train, y_train, 
        epochs=1, 
        batch_size=32, 
        verbose=1)

w_after,b_after = models.layers[0].set_weights()
delta_w = w_after -w_before


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tinymlgen import port

# Data
sample_data = 10000
x_train = np.random.rand(sample_data, 2)
y_train = x_train[:, 0] + x_train[:, 1]

# Build model
model=Sequential()
model.add(Dense(2,input_shape=(2,),activation='linear',name='Input_Layer'))
model.add(Dense(8,activation='linear',name='Hidden_Layer_1'))
model.add(Dense(8,activation='linear',name='Hidden_Layer_2'))
model.add(Dense(1,name='Output_Layer'))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Train model
model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=2)

#Changing to a TF lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

c_code = port(model, pretty_print=True) 
with open("tiny_ADD_model.h", "w") as f: 
    f.write(c_code)





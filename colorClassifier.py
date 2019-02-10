import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time

colorData = pd.read_json('colorData.json', orient='records')
labels = colorData.pop('label')
uid = colorData.pop('uid')

labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]

#print(colorData.head())
#print(colorData.info())

labels = pd.get_dummies(labels)
colorData = colorData/255

colorData = np.array(colorData)
labels = np.array(labels)

s=np.arange(colorData.shape[0])
np.random.shuffle(s)
colorData=colorData[s]
labels=labels[s]

model = keras.Sequential()
model.add(keras.layers.Dense(32, input_shape = (3,), activation = 'sigmoid'))
model.add(keras.layers.Dense(9, activation = 'softmax'))
#print(model.summary())
learningRate = 0.25
optimizer = keras.optimizers.SGD(learningRate)

tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))

model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])

model.fit(colorData, labels, validation_split = 0.2, batch_size=50, epochs=50, verbose=1, shuffle = True, callbacks = [tensorboard])

accuracy = model.evaluate(colorData, labels, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])






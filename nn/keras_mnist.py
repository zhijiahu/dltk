
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='path to output loss/accuracy plot')
args = vars(ap.parse_args())

print('[INFO] loading MNIST (full) dataset')
dataset = datasets.fetch_mldata('MNIST Original')

# normalize
data = dataset.data.astype('float') / 255.0
(train_x, test_x, train_y, test_y) = train_test_split(data, dataset.target, test_size=0.25)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# build model
model = Sequential() # feed forward, output of prev layer feeding into next
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax')) # get normalized class probabilities

print('[INFO] training network...')
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
H = model.fit(train_x, train_y, validation_data=(test_x, test_y),
              epochs=100, batch_size=128)







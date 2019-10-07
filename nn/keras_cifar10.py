
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='path to output loss/accuracy plot')
args = vars(ap.parse_args())

print('[INFO] loading CIFAR-10 dataset')
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
train_x = train_x.astype('float') / 255.0
test_x = test_x.astype('float') / 255.0
train_x = train_x.reshape((train_x.shape[0], 3072)) # 32 X 32 X 3
test_x = test_x.reshape((test_x.shape[0], 3072))

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

label_names = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# build model
model = Sequential() # feed forward, output of prev layer feeding into next
model.add(Dense(1024, input_shape=(3072,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax')) # get normalized class probabilities

print('[INFO] training network...')
sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
H = model.fit(train_x, train_y, validation_data=(test_x, test_y),
              epochs=100, batch_size=32)

print('[INFO] evaluating network...')
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=label_names))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

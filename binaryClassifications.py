from keras.datasets import imdb
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from keras import layers
num_words = 1000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
def normalizeData(data, size = num_words):
 res = np.zeros((len(data), size))
 for i in range(0, len(data)):
  for j in data[i]:
   res[i][j] = 1
 return res
x_train = normalizeData(train_data)
x_test = normalizeData(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(num_words,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
number_of_epochs=10
history = model.fit(x_train, y_train, epochs=number_of_epochs, batch_size=512, validation_data=(x_test, y_test))
history_data = history.history
p = model.predict(x_test)
loss_values = history_data['loss']
val_loss_values = history_data['val_loss']
accuracy = history_data['accuracy']
val_accuracy = history_data['val_accuracy']
epochs = range(1, number_of_epochs + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs, accuracy, 'bo', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import Dense, Activation, LSTM, Dropout
from keras.layers import Embedding

import numpy as np
import csv
import os
from random import shuffle


class RNN:
    def __init__(self):
        self._model = Sequential()
        self._model.add(LSTM(32, return_sequences=True,
                             input_shape=(1, 50)))  # returns a sequence of vectors of dimension 32
        self._model.add(LSTM(32,
                             return_sequences=True))  # returns a sequence of vectors of dimension 32
        self._model.add(LSTM(32))  # return a single vector of dimension 32
        self._model.add(Dense(4, activation='softmax'))
        self._model.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        current_path = os.getcwd()
        data_labels = []
        for category in range(4):
            filenames = os.listdir("trainingdata/%s" % category)
            for filename in filenames:
                with open("%s/trainingdata/%s/%s" % (current_path, category, filename),
                          newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    tmp_list = []
                    for row in spamreader:
                        tmp_list.append(float(row[0]))
                    data_labels.append((tmp_list, category))
        shuffle(data_labels)
        data = []
        labels = []
        for tuple in data_labels:
            data.append([tuple[0]])
            labels.append(tuple[1])
        data = np.array(data)
        labels = np.array(labels)
        labels = np_utils.to_categorical(labels, 4)
        self._model.fit(data, labels, epochs=5, batch_size=2)

    def predict(self, data):
        category = list(self._model.predict(np.array(data))[0])
        result = category.index(max(category))
        return result


r = RNN()

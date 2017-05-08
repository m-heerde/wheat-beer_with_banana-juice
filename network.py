from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.optimizers import RMSprop

from keras.layers import Dense, LSTM
import numpy as np
import csv
import os
from random import shuffle


class RNN:
    def __init__(self):
        try:
            self._model = load_model("model.h5")
        except OSError:
            self._model = Sequential()
            self._model.add(LSTM(32, return_sequences=True,
                                 input_shape=(
                                     40, 1)))  # returns a sequence of vectors of dimension 32
            self._model.add(LSTM(32,
                                 return_sequences=True))  # returns a sequence of vectors of dimension 32
            self._model.add(LSTM(32))  # return a single vector of dimension 32
            self._model.add(Dense(4, activation='softmax'))
            optimizer = RMSprop()
            self._model.compile(optimizer=optimizer,
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
                            tmp_list.append([float(row[0])])
                        for i in range(10):
                            j = -1 * (10 - i)
                            lst = tmp_list[i:j]
                            data_labels.append((lst, category))
            shuffle(data_labels)
            data = []
            labels = []
            for tuple in data_labels:
                data.append(tuple[0])
                labels.append(tuple[1])
            data = np.array(data)
            labels = np.array(labels)
            labels = np_utils.to_categorical(labels, 4)
            self._model.fit(data, labels, epochs=50, batch_size=50)
            self._model.save("model.h5")


    def predict(self, data):
        try:
            data = [[float(val)] for val in data['data']]
        except TypeError:
            data = [[val] for val in data]
        category = list(self._model.predict(np.array([data]))[0])
        result = category.index(max(category))
        return result


if __name__ == '__main__':

    bsp_0 = [76.0, 77.0, 77.0, 77.0, 76.0, 77.0, 77.0, 77.0, 77.0, 77.0,
             77.0, 77.0, 78.0, 77.0, 77.0, 78.0, 78.0, 78.0, 79.0, 79.0,
             79.0, 80.0, 79.0, 80.0, 80.0, 80.0, 82.0, 82.0, 82.0, 81.0,
             82.0, 83.0, 85.0, 85.0, 85.0, 84.0, 85.0, 86.0, 85.0, 85.0]
    bsp_1 = [79.0, 79.0, 80.0, 82.0, 81.0, 82.0, 83.0, 82.0, 83.0, 82.0,
             83.0, 83.0, 83.0, 84.0, 83.0, 83.0, 84.0, 84.0, 84.0, 85.0,
             85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0,
             85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 84.0, 84.0, 85.0, 85.0]

    bsp_2 = [128, 127, 127, 128, 128, 130, 132, 130, 131, 131, 132, 132, 134, 132, 134,
             136, 137, 138, 136, 137, 141, 139, 139, 140, 141, 143, 144, 144, 145, 144,
             148, 146, 148, 146, 147, 150, 151, 150, 151, 150, 153, 153, 152, 153, 152,
             154, 154, 154, 156, 155]
    r = RNN()
    print(r.predict(bsp_0))
    print(r.predict(bsp_1))
    for i in range(10):
        j = -1 * (10 - i)
        print(r.predict(bsp_2[i:j]))

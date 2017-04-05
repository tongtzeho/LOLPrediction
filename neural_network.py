from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from keras.optimizers import SGD, Adam, RMSprop

import gzip
import sys
from six.moves import cPickle

from fetcher import *

batch_size = 256
nb_classes = 2
nb_epoch = 50

champion_dict = fetch_champion_dict("champion.json")
champion_num = len(champion_dict)

#X_train, y_train, X_test, y_test = read_file_one_side_old('AramDataSet38W.txt', 'ChampionList624.txt', 134)
#X_train, y_train, X_test, y_test = read_file_one_side('AramDataSet624.txt', 'ChampionList624.txt', 134)
#X_train, y_train, X_test, y_test = fetch_one_side_tgp('arurf', champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'NORMAL', 'CLASSIC', ('1490371200000', '1800371200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'RANKED_SOLO_5x5', 'CLASSIC', ('1490371200000', '1800371200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1490371200000', '1800371200000'), champion_dict)

#X_train, y_train, X_test, y_test = read_file_both_sides_old('AramDataSet38W.txt', 'ChampionList624.txt', 134)
X_train, y_train, X_test, y_test = read_file_both_sides('AramDataSet624.txt', 'ChampionList624.txt', 134)
#X_train, y_train, X_test, y_test = fetch_both_sides_tgp('arurf', champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'NORMAL', 'CLASSIC', ('1490371200000', '1800371200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'RANKED_SOLO_5x5', 'CLASSIC', ('1490371200000', '1800371200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1490371200000', '1800371200000'), champion_dict)
		
X_train = X_train.astype('int8')
X_test = X_test.astype('int8')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(500, input_dim = champion_num*2, init='uniform'))
model.add(Activation('sigmoid'))
#model.add(Dense(300))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(400))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#model.save('aram_neuralnetwork.h5')

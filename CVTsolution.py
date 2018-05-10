#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Activation, advanced_activations, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras import regularizers, optimizers
from keras import backend as K

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# from imutils import paths
from PIL import Image

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import shutil
import json
import sys
import cv2
import os

np.random.seed(25)

def preprocessing(input_dir):
  path = "/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/" + input_dir + "/"
  dirs = os.listdir(path)
  for item in dirs:
    if ((os.path.isfile(path+item)) and (item[-3:] == 'jpg')):
      im = Image.open(path+item)
      f, e = os.path.splitext(path+item)
      imResize = im.resize((150,150), Image.ANTIALIAS)
      imResize.save(f + '.jpg', 'JPEG', quality=90)
      
def validation_train_split(input_dir):
  path = "/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/" + input_dir + "/"
  train_path = "/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/train/" + input_dir
  validation_path = "/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/validation/" + input_dir

  files = os.listdir(path)
  for f in files:
    if (f[-3:] == 'jpg'):
      if (np.random.random(1) < 0.2):
        shutil.move(path + f, validation_path + '/'+ f)
      else:
        shutil.move(path + f, train_path + '/'+ f)
    
def load_data(brands):
  data = []
  labels = []
  colors = {"green": ["ЗЕЛЕНЫЙ МЕТАЛЛИК", "ЗЕЛЕНЫЙ", "СИНЕ-ЗЕЛЕНЫЙ", "ТЕМНО-ЗЕЛЕНЫЙ"],
            "red": ["КРАСНЫЙ", "БОРДОВЫЙ МЕТАЛЛИК", "СЕРО-ЗЕЛЕНЫЙ ПЕРЛАМУТРОВЫЙ", "ЧЕРНЫЙ ПЕРЛАМУТРОВЫЙ", "СЕРО-ПЕРЛАМУТРОВЫЙ", "ВИШНЕВЫЙ", "ГОЛУБОЙ ПЕРЛАМУТРОВЫЙ", "ОРАНЖЕВЫЙ ПЕРЛАМУТРОВЫЙ", "ПЕСОЧНО-ПЕРЛАМУТРОВЫЙ", "КРАСНЫЙ МЕТАЛЛИК", "БЕЛЫЙ ПЕРЛАМУТРОВЫЙ", "ТЕМНО-КРАСНЫЙ", "ВИШНЕВЫЙ МЕТАЛЛИК"],
            "black": ["ЧЕРНЫЙ", "ЧЕРНЫЙ МЕТАЛЛИК", "КОМБИНИРОВАННЫЙ"],
            "blue": ["ТЕМНО-СИНИЙ", "СИНИЙ МЕТАЛЛИК", "ТЕМНО-СИНИЙ МЕТАЛЛИК", "СИНИЙ", "ГОЛУБОЙ МЕТАЛЛИК", "ЗОЛОТИСТЫЙ", "ФИОЛЕТОВЫЙ"],
            "brown": ["КОРИЧНЕВЫЙ МЕТАЛЛИК", "КОРИЧНЕВЫЙ", "ТЕМНО-КОРИЧНЕВЫЙ"],
            "yellow": ["БЕЖЕВЫЙ МЕТАЛЛИК", "СВЕТЛО-БЕЖЕВЫЙ", "ЗОЛОТИСТЫЙ МЕТАЛЛИК", "БЕЖЕВЫЙ", "САХАРА", "ОРАНЖЕВЫЙ"],
            "gray": ["СЕРЫЙ МЕТАЛЛИК", "СЕРЫЙ", "ТЕМНО-СЕРЫЙ МЕТАЛЛИК", "БЕЛО-ДЫМЧАТЫЙ", "СВЕТЛО-СЕРЫЙ", "СВЕТЛО-СЕРЫЙ МЕТАЛЛИК"],
            "white": ["БЕЛЫЙ МЕТАЛЛИК", "БЕЛЫЙ", "ЖЕМЧУЖНО-БЕЛЫЙ"],
            "silver": ["СЕРЕБРИСТЫЙ", "СЕРЕБРИСТЫЙ МЕТАЛЛИК"]}

  for brand in brands:
    input_dir = brand
    path = "/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/" + input_dir + "/"
  
    files = os.listdir(path)

    for f in files:
      if ((os.path.isfile(path+f)) and (f[-3:] == 'jpg')):
        image = cv2.imread(path+f)
        image = cv2.resize(image, (96, 96))
        image = img_to_array(image)
        data.append(image)
      else:
        with open(path+f) as json_file:
          json_data = json.load(json_file)
          for key, value in colors.items():
            if ((json_data["color"] in value) and (json_data["model"].split(" ")==["TOYOTA", "RAV", "4"] or json_data["model"].split(" ")==["TOYOTA", "RAV4"])):
              label = [key, "TOYOTA RAV4"]
            elif (json_data["color"] in value):
              label = [key, json_data["model"]]
          labels.append(label)
  print("labels: ")
  print(labels)
  data = np.array(data, dtype="float") / 255.0
  labels = np.array(labels)
  
  mlb = MultiLabelBinarizer()
  labels = mlb.fit_transform(labels)
  print(mlb.classes_)
  f = open("mlb.pickle", "wb")
  f.write(pickle.dumps(mlb))
  f.close()

  return data, labels
  
def cnn_model_one():
  model = Sequential()

  # main layers
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(32, (3, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(Conv2D(32, 3))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(BatchNormalization())
  model.add(Dropout(0.3))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(64, (3, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(Conv2D(64, (3, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(64, (3, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(Conv2D(64, (3, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(128, (3, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(Conv2D(128, (3, 3)))
  # model.add(Activation('relu'))
  model.add(advanced_activations.PReLU())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  # dense layers
  model.add(Flatten())
  model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
  model.add(advanced_activations.PReLU())
  model.add(Dense(32))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.6))
  model.add(Dense(30))
  model.add(Activation('softmax'))

  return model

def cnn_model_two():
  model = Sequential()

  model.add(Conv2D(32, (3, 3), padding='same', input_shape=(150, 150, 3), activation='relu'))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
    
  model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(30, activation='softmax'))

  return model

def cnn_model_three(outputActivation='softmax'):
  model = Sequential()

  model.add(Conv2D(32, (3, 3), padding='same', input_shape=(96, 96, 3)))
  model.add(Activation('relu'))
  model.add(BatchNormalization(axis=-1))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding="same", input_shape=(96, 96, 3)))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=-1))
  model.add(Conv2D(64, (3, 3), padding="same", input_shape=(96, 96, 3)))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=-1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  
  model.add(Conv2D(128, (3, 3), padding="same", input_shape=(96, 96, 3)))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=-1))
  model.add(Conv2D(128, (3, 3), padding="same", input_shape=(96, 96, 3)))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=-1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
 
  model.add(Dense(39))
  model.add(Activation(outputActivation))
  
  return model

if __name__ == '__main__':
  brands = ["AUDI A6", "BMW 520", "BMW X5", "CHEVROLET CRUZE", "DAEWOO NEXIA", "HYUNDAI ACCENT", "KIA RIO", "KIA SPORTAGE", "LEXUS LX 570", "LEXUS RX 300", "MERCEDES-BENZ", "MITSUBISHI OUTLANDER", "NISSAN ALMERA", "NISSAN JUKE", "NISSAN PATROL", "NISSAN QASHQAI", "NISSAN TEANA", "RENAULT DUSTER", "RENAULT LOGAN", "SUBARU FORESTER", "TOYOTA CAMRY", "TOYOTA COROLLA", "TOYOTA HIGHLANDER", "TOYOTA LAND CRUISER 200", "TOYOTA LAND CRUISER PRADO", "TOYOTA RAV4", "VOLKSWAGEN GOLF", "VOLKSWAGEN PASSAT", "VOLKSWAGEN POLO", "VOLKSWAGEN TOUAREG"]

  callbacks = [EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max'), ModelCheckpoint('/home/otaniyoz/Documents/projects/ABCHack2018/checkpoint.h5', monitor='val_acc', save_best_only=True, mode='max', verbose=0)]

  for brand in brands:
    preprocessing(brand)
    # validation_train_split(brand)

  data, labels = load_data(brands)

  model = cnn_model_three(outputActivation="sigmoid")
  # model = cnn_model_two()
  # model = cnn_model_one()

  # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  # model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
  # model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
  model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
  
  model.summary()

  batch_size = 16

  (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
  
  train_datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, fill_mode='nearest', horizontal_flip=True)
  # test_datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rescale=1./255, rotation_range=0.45, horizontal_flip=True)

  # train_generator = train_datagen.flow_from_directory('/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/train', target_size=(96, 96), batch_size=batch_size, class_mode='categorical')
  # validation_generator = test_datagen.flow_from_directory('/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/validation', target_size=(96, 96), batch_size=batch_size, class_mode='categorical')

  # history = model.fit_generator(train_generator, steps_per_epoch=2000, epochs=75, validation_data=validation_generator, validation_steps=800, verbose=1, callbacks=callbacks)
  history = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size = batch_size), validation_data = (testX, testY), steps_per_epoch=len(trainX) // batch_size, epochs=75, verbose=1)# , callbacks=callbacks)
  model.save('CVTsolution.model')
  
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()
  plt.savefig('graph.png')

#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from PIL import Image

import shutil, os, sys
import numpy as np

np.random.seed(25)

brands = ["AUDI A6", "BMW 520", "BMW X5", "CHEVROLET CRUZE", "DAEWOO NEXIA", "HYUNDAI ACCENT", "KIA RIO", "KIA SPORTAGE", "LEXUS LX 570", "LEXUS RX 300", "MERCEDES-BENZ", "MITSUBISHI OUTLANDER", "NISSAN ALMERA", "NISSAN JUKE", "NISSAN PATROL", "NISSAN QASHQAI", "NISSAN TEANA", "RENAULT DUSTER", "RENAULT LOGAN", "SUBARU FORESTER", "TOYOTA CAMRY", "TOYOTA COROLLA", "TOYOTA HIGHLANDER", "TOYOTA LAND CRUISER 200", "TOYOTA LAND CRUISER PRADO", "TOYOTA RAV4", "VOLKSWAGEN GOLF", "VOLKSWAGEN PASSAT", "VOLKSWAGEN POLO", "VOLKSWAGEN TOUAREG"]

callbacks = [EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max'), ModelCheckpoint('/home/otaniyoz/Documents/projects/ABCHack2018/model.h5', monitor='val_acc', save_best_only=True, mode='max', verbose=0)]
# LearningRateScheduler(lr_schedule), 

def lr_schedule(epoch):
  return 0.01 * (0.1 ** int(epoch / 10))

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

def cnn_model_one():
  model = Sequential()

  # main layers
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(128, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  # dense layers
  model.add(Flatten())
  model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
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

for brand in brands:
  preprocessing(brand)
  validation_train_split(brand)

model = cnn_model_one()
# model = cnn_model_two()

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

model.summary()

batch_size = 16

train_datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rescale=1./255, rotation_range=0.45, shear_range=0.2, zoom_range=0.5, channel_shift_range=0.5, fill_mode='nearest', horizontal_flip=True)
test_datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rescale=1./255, rotation_range=0.45, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory('/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/train', target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('/home/otaniyoz/Documents/projects/ABCHack2018/Hackathon_for_training/validation', target_size=(150, 150), batch_size=batch_size, class_mode='categorical')

model.fit_generator(train_generator, steps_per_epoch=2000, epochs=25, validation_data=validation_generator, validation_steps=800, verbose=1, callbacks=callbacks)
model.save_weights('CVTsolution_baseline.h5')

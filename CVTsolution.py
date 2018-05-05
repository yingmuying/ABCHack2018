from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

im = Image.open('/home/otaniyoz/Documents/projects/ABCHack2018/for ABC Hackathon_for_training/AUDI A6/114104dadd9a30058af3f57ab0731e15.jpg')
width, height = im.size
print(width, height)

model = models.Sequential()

# main layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape= (3, 150, 150)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# dense layers
model.add(Flatten())
model.add(Dense(64), activation='relu')
model.add(Dropout(0.5))
model.add(Dense(10), activation='softmax')

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/train', target_size=(150, 150), batch_size=batch_size, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('data/validation', target_size=(150, 150), batch_size=batch_size, class_mode='binary')

model.fit_generator(train_generator, steps_per_epoch=2000, epochs=50, validation_data=validation_generator, validation_steps=800)
model.save_weigths('CVTsolution_baseline.h5')

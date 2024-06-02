# -*- coding: utf-8 -*-
"""
Created on 2020-10-21

@author: ganjar@dongguk.edu
"""


import os
import tensorflow as tf
import matplotlib.pyplot as plt

data_path = "dataset/"
base_dir = os.path.join(data_path)
IMAGE_SIZE = 124 #model input size
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        validation_split=0.2)
train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        subset='training')
val_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        subset='validation')
for image_batch,label_batch in train_generator:
    break
image_batch.shape, label_batch.shape
#to see sample image
plt.imshow(image_batch[0])
plt.show()
#to see sample label
label_batch[0]
#---
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
num_classes=2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=IMG_SHAPE))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(28, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(num_classes))
model.add(tf.keras.layers.Activation('softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
epochs = 20
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator)
#save the model
model.save('CNN_cat_dog.model')
'''
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#save the model
'''
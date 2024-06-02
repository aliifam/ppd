

import tensorflow as tf
import numpy as np


IMAGE_SIZE = 124
loadedmodel = tf.keras.models.load_model('CNN_cat_dog.model')

img = tf.keras.preprocessing.image.load_img('garfield.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = loadedmodel.predict(img_array, steps=1)
int_result = np.argmax(predictions[0])

if(int_result==0):
    decision='Cat'
else:
    decision='Dog'


print("This image most likely belongs to", decision, " with a percent confidence", 100 * np.max(predictions[0]))
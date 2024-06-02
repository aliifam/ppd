"""
Created on 2020-11-25
@author: Dr. Ganjar Alfian
email : ganjar@dongguk.edu
for teaching purpose only.
"""


import os
from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)
os.path.dirname(__file__)
            
#load first page           
@app.route("/image/")
def index():
    return render_template('index.html')


@app.route('/image/result/', methods=["POST"])
def prediction_result():
    #start uploading file or image from client
    uploaded_file = request.files['image']
    file_name = uploaded_file.filename
    #save the file inside static folder
    UPLOAD_FOLDER = os.path.abspath('static/')
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(UPLOAD_FOLDER, file_name))
            

    #end uploading
    
    #start predicting the image
    IMAGE_SIZE = 124
    loadedmodel = tf.keras.models.load_model('CNN_cat_dog.model')
    
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = loadedmodel.predict(img_array, steps=1)
    int_result = np.argmax(predictions[0])
    
    if(int_result==0):
        decision='Cat'
    else:
        decision='Dog'
    
    conf = 100 * np.max(predictions[0])
    #print("This image most likely belongs to", decision, " with a percent confidence", )
    #return the output and load result.html
    return render_template('result.html', status=decision, confidence = conf, upload_name=file_name)

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port='5003')
    app.run()
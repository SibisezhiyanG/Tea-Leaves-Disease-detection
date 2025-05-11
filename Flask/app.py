#import re
import numpy as np
import os
from flask import Flask, app,request,render_template, redirect, url_for,jsonify
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img,img_to_array


#Loading the model
modeln=load_model(r"D:\TEA LEAVES\Flask\vgg-16-Tea-leaves-disease-model.h5")


app=Flask(__name__)

#default home page or route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index')
def inde1():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template("about.html")



@app.route('/teahome')
def teahome():
    return render_template('teahome.html')

@app.route('/teapred')
def teapred():
    return render_template('teapred.html')

@app.route('/tearesult',methods=["GET","POST"])
def nres():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__) #getting the current path i.e where app.py is present
        #print("current path",basepath)
        filepath=os.path.join(basepath,'uploads',f.filename) #from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        #print("upload folder is",filepath)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)#img to array
        x=np.expand_dims(x,axis=0)#used for adding one more dimension
        #print(x)
        img_data=preprocess_input(x)
        prediction=np.argmax(modeln.predict(img_data))

        index=['Anthracnose: Anthracnose is a common plant disease caused by various species of fungi in the genus Colletotrichum, as well as some other genera. It affects a wide range of plants, including trees, shrubs, fruits, and vegetables. Anthracnose typically manifests as dark, sunken lesions on leaves, stems, fruits, or flowers.','algal leaf  40%: Algal leaf spot, also known as Cephaleuros virescens, is a plant-parasitic green algae that infects tea plants, causing orange, rust-colored, dense silky tufts to appear on the leaf surface.' ,"bird eye spot: Bird's eye spot disease, also known as bird's eye rot or bird's eye spot, is a fungal disease that commonly affects various fruits, including apples, pears, and stone fruits like cherries and peaches.","brown blight: Brown blight disease can refer to several different plant diseases, depending on the context. One common meaning of 'brown blight disease' refers to a fungal disease affecting various tree species, particularly oaks and other hardwoods. This disease is often caused by the fungus Phytophthora cinnamomi and can result in the death of infected trees.  Symptoms of brown blight disease in trees typically include wilting, browning, and dieback of leaves and branches.","gray light: Gray leaf spot (GLS) is a fungal disease that affects maize, also known as corn. It's caused by two pathogens, Cercospora zeae-maydis and Cercospora zeina. GLS is one of the most significant yield-limiting diseases of corn worldwide.",'healthy',"red leaf spot: Red leaf spot is a disease that occurs on creeping bentgrass during warm and wet weather in the spring, summer, or fall. Red leaf spot is a 'Helminthosporium' disease, which is a complex of diseases caused by fungi that produce large, cigar-shaped spores.",'white spot: The fungus Ceratocystis paradoxa causes white leaf spot, black rot, base or but rot and soft rot or water blisters. White leaf spots are yellow to brown and several centimetres long. Later they dry to become papery and straw coloured.']
        nresult = str(index[prediction])
        nresult        
        
        return render_template('teapred.html',prediction=nresult)
        



""" Running our application """
if __name__ == "__main__":
    app.run(debug =False, port = 8080)
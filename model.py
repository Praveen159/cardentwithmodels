# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
#from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array,load_img
#import tensorflow as tf
Damage=load_model("C:/Users/hi/POC/Model_Fine/Damage/ep_5_st100_bs8_l0.01r.h5")
#graph=tf.get_default_graph()
#Damage._make_predict_function()
Severe=load_model("C:/Users/hi/POC/Model_Fine/Severity/sever_model_epoch2_bs100.h5")
Severe._make_predict_function()
#Damage=load_model("./models/epoch_2.h5")
#Severe=load_model("./models/sever_model_epoch2.h5")

#import PIL

def predic(img_path):
    img_256=load_img(img_path,target_size=(256,256))
    x=img_to_array(img_256)
    x=x.reshape((1,)+x.shape)/255
    #print("YESSSSSSSSSSSSSSS")
    predi_damage=Damage.predict(x)
    
    #with graph.as_default() :
        #predi_damage=Damage.predict(x)
    #print("INDIAAAAAAAAAAAAAAAAAA")
    if predi_damage<0.8:
        damag="Car Got Damaged"
        #print("NOOOOOOOOOOOOOOOOOOO")
        predi_severe=Severe.predict(x)
        pred_severe_1=np.argmax(predi_severe,axis=1)
        dic_severe={0:'Minor',1:'Major',2:'Severe'}
        key=pred_severe_1[0]
        severeity_pred=dic_severe[key]
        
        return severeity_pred,damag
    else:
        c="Please Submit  damaged  CAR Image"
        return c

#output=predic("D:/DS/Train_Images/CarDentImages/CarDentImages1.jpg")
    

	  

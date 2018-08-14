# -*- coding: utf-8 -*-
#from keras.applications.vgg16 import VGG16
from keras import applications
from keras.preprocessing.image import img_to_array,load_img
#from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras import backend as K
import tensorflow as tf
img_width=224
img_height=224
model1 = applications.VGG16(weights='imagenet')
#K.clear_session()

def pred_car(img):
    im=load_img(img,target_size=(224,224))
    x=img_to_array(im)
    x=x.reshape((1,)+x.shape)
    x = preprocess_input(x)
    model1._make_predict_function()
    pred=model1.predict(x)
    K.clear_session()
    tf.reset_default_graph()
    tf.contrib.keras.backend.clear_session()
    label=decode_predictions(pred)
    label = label[0][0]
    #print('%s (%.2f%%)' % (label[1], label[2]*100))
    return label[1],label[2]
    
#a=pred_car("D:/DS/Train_Images/CarDentImages/CarDentImages1.jpg")    



#x.shape







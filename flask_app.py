from flask import Flask
from flask import render_template,request
#from keras.models import load_model
from os.path import dirname,realpath

from werkzeug import secure_filename
#from keras.preprocessing.image import load_img
#import numpy as np
import os
#import tensorflow as tf
#tf.__version__
import model
import car
#App will crash bevause of gunicorn set up in procfile we should be 
#careful with that
#requirements and conda requirements add all the packages
app=Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
main_folder=os.path.join(PROJECT_ROOT,"Images_folder")
print(main_folder)
app.config['UPLOAD_FOLDER'] = main_folder
@app.route('/')
def input():
    return render_template("input.html")

@app.route('/output',methods=['GET','POST'])
def image():
    if request.method=='POST':
        f=request.files['file']
        fil=secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], fil))
        
        path=os.path.join(app.config['UPLOAD_FOLDER'], fil)
        output1=car.pred_car(path)
        if output1 :
            output=model.predic(path)
        else:
            output="Please submit Car Image"
            
        
        #list_passed=["Please","Submit"]
        return render_template("output.html",result=output)
        
if __name__=='__main__':
    app.run(debug=False,threaded=False)
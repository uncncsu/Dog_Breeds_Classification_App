from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import heapq

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
#from keras.applications.xception import Xception
#from keras.applications.xception import preprocess_input as preprocess_input_xception, decode_predictions as decode_predictions_xception
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


#SQLALCHEMY
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine,MetaData

from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy


#Dog array
dog_names = ['Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 
'Akita', 'Alaskan_malamute', 'American_eskimo_dog', 'American_foxhound', 
'American_staffordshire_terrier', 'American_water_spaniel', 
'Anatolian_shepherd_dog', 'Australian_cattle_dog', 
'Australian_shepherd', 'Australian_terrier', 'Basenji', 'Basset_hound', 
'Beagle', 'Bearded_collie', 'Beauceron', 'Bedlington_terrier', 
'Belgian_malinois', 'Belgian_sheepdog', 'Belgian_tervuren', 'Bernese_mountain_dog', 'Bichon_frise', 
'Black_and_tan_coonhound', 'Black_russian_terrier', 'Bloodhound', 'Bluetick_coonhound', 'Border_collie',
'Border_terrier', 'Borzoi', 'Boston_terrier', 'Bouvier_des_flandres', 'Boxer', 'Boykin_spaniel', 'Briard', 
'Brittany', 'Brussels_griffon', 'Bull_terrier', 'Bulldog', 'Bullmastiff', 'Cairn_terrier', 'Canaan_dog', 
'Cane_corso', 'Cardigan_welsh_corgi', 'Cavalier_king_charles_spaniel', 'Chesapeake_bay_retriever', 'Chihuahua', 
'Chinese_crested', 'Chinese_shar-pei', 'Chow_chow', 'Clumber_spaniel', 'Cocker_spaniel', 'Collie', 'Curly-coated_retriever', 
'Dachshund', 'Dalmatian', 'Dandie_dinmont_terrier', 'Doberman_pinscher', 'Dogue_de_bordeaux', 'English_cocker_spaniel', 
'English_setter', 'English_springer_spaniel', 'English_toy_spaniel', 'Entlebucher_mountain_dog', 'Field_spaniel', 'Finnish_spitz', 
'Flat-coated_retriever', 'French_bulldog', 'German_pinscher', 'German_shepherd_dog', 'German_shorthaired_pointer', 
'German_wirehaired_pointer', 'Giant_schnauzer', 'Glen_of_imaal_terrier', 'Golden_retriever', 'Gordon_setter', 'Great_dane', 
'Great_pyrenees', 'Greater_swiss_mountain_dog', 'Greyhound', 'Havanese', 'Ibizan_hound', 'Icelandic_sheepdog', 
'Irish_red_and_white_setter', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 
'Japanese_chin', 'Keeshond', 'Kerry_blue_terrier', 'Komondor', 'Kuvasz', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberger', 
'Lhasa_apso', 'Lowchen', 'Maltese', 'Manchester_terrier', 'Mastiff', 'Miniature_schnauzer', 'Neapolitan_mastiff', 'Newfoundland', 
'Norfolk_terrier', 'Norwegian_buhund', 'Norwegian_elkhound', 'Norwegian_lundehund', 'Norwich_terrier', 
'Nova_scotia_duck_tolling_retriever', 'Old_english_sheepdog', 'Otterhound', 'Papillon', 'Parson_russell_terrier', 'Pekingese', 
'Pembroke_welsh_corgi', 'Petit_basset_griffon_vendeen', 'Pharaoh_hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle', 
'Portuguese_water_dog', 'Saint_bernard', 'Silky_terrier', 'Smooth_fox_terrier', 'Tibetan_mastiff', 'Welsh_springer_spaniel', 
'Wirehaired_pointing_griffon', 'Xoloitzcuintli', 'Yorkshire_terrier']

# Define a flask app
app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "postgres://kogocmkjkapxrj:38623a2b98c06e0a7ac66f4d990e7a3a563531123455fd9dac34a85e673926c4@ec2-174-129-210-249.compute-1.amazonaws.com:5432/df0ceqcsqb5gl6"
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '')
db = SQLAlchemy(app)

# reflect our database and table, save reference to table
Base = automap_base()
Base.prepare(db.engine, reflect=True, schema="public")
characteristic = Base.classes.breed_characterz

# Point to our saved model, load the model
MODEL_PATH = 'models/dogClassification.h5'
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')    

# Function to process image as necessary for Xception model base
def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# Function to load, process, and make a prediction
def model_predict(img_path, model):
    # loads image and process
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    test_image = np.expand_dims(x, axis=0)
    test_image = extract_Xception(test_image)
    # make prediction with our model
    predicted_vector = model.predict(test_image)
    # associate highest probability with dog breed
    preds = dog_names[np.argmax(predicted_vector)]
    return preds

def model_predict2(img_path, model):
    # loads image and process
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    test_image = np.expand_dims(x, axis=0)
    test_image = extract_Xception(test_image)
    # make prediction with our model
    predicted_vector = model.predict(test_image)
    # associate highest probability with dog breed
    top2=heapq.nlargest(2, range(len(predicted_vector[0])),key=predicted_vector[0].__getitem__)
    preds2 = dog_names[top2[1]]
    return preds2


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/id', methods=['GET'])
def predict():
    # Prediction Page
    return render_template('id.html')

@app.route('/data', methods=['GET'])
def data():
    # Data Accuracy Page
    return render_template('data.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to temporary /uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Call our prediction function
        preds = model_predict(file_path, model)
        breed = str(preds)

        return breed
    return None

@app.route('/predict2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to temporary /uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Call our prediction function
        preds2 = model_predict2(file_path, model)
        breed2 = str(preds2)
        return breed2
    return None

@app.route("/id/predict/<breed>")
def get_data_results(breed):
    """Return the data for a given year."""
    sel = [
        characteristic.id,
        characteristic.BreedName,
        characteristic.Group1,
        characteristic.Group2,
        characteristic.MaleWtKg,
        characteristic.Temperment,
        characteristic.AvgPupPrice,
        characteristic.Intelligence,
        characteristic.Watchdog,
        characteristic.PopularityUS2017
    ]

    results = db.session.query(*sel).filter(characteristic.BreedName == breed).all()

    # Create a dictionary entry for each row of metadata information
    breeds_list = []
    
    
    for result in results:
        json_breed_characterz = {}

        #json_breed_characterz["ID"] = result[0]
        #json_breed_characterz["Breed Name"] = result[1]
        json_breed_characterz["Group 1"] = result[2]
        json_breed_characterz["Group 2"] = result[3]
        json_breed_characterz["Average Male Weight (kg)"] = result[4]
        json_breed_characterz["Temperment"] = result[5]
        json_breed_characterz["Average Puppy Price"] = result[6]
        json_breed_characterz["Intelligence"] = result[7]
        json_breed_characterz["Watchdog"] = result[8]
        json_breed_characterz["Popularity"] = result[9]
        breeds_list.append(json_breed_characterz)

    print(json_breed_characterz)
    return jsonify(breeds_list)
    print(json_breed_characterz)


if __name__ == '__main__':
    app.run()
    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 4001), app)
    # http_server.serve_forever()
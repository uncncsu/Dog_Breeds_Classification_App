	import os
    import io
    import numpy as np
    
    import keras
    from keras.preprocessing import image
    from keras.preprocessing.image import img_to_array
    from keras.applications.xception import (
        Xception, preprocess_input, decode_predictions)
    from keras import backend as K
    
    from flask import Flask, request, redirect, url_for, jsonify, render_template
    
    from werkzeug.utils import secure_filename
    
    UPLOAD_FOLDER = 'static'
    ALLOWED_EXTENSIONS = set(['h5', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
    
    app = Flask(__name__)
    app.config['model'] = UPLOAD_FOLDER
    
    model = None
    
    #FILE VALIDATION
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    
    def load_model():
        Model=Load_model.(path/to/model) 
    
    
    
    
    
    load_model()
    
    
    def prepare_image(img):
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        # return the processed image
        return img_path
    
    
    def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    
    return render_template("index.html",display=display,imagefile=imagefile)
    
    
    
    if __name__ == "__main__":
        app.run(debug=True)

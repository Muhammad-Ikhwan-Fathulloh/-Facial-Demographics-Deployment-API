import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Define a flask app
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Model saved with Keras model.save()
MODEL_PATH = 'models/a_g1.h5'

# Load your trained model
new_model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

def loadImage(filepath):
    test_img = image.load_img(filepath, target_size=(198, 198))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img /= 255
    return test_img

def model_predict(img_path):
    global new_model    
    age_pred, gender_pred = new_model.predict(loadImage(img_path))
    img = image.load_img(img_path)                        

    max=-1
    count=0
    xx = list(age_pred[0])
    for i in age_pred[0]:
        if i>max:
            max = i
            temp = count
        count+=1

    age_categories = ['0-24 yrs old', '25-49 yrs old', '50-74 yrs old', '75-99 yrs old', '91-124 yrs old']
    age = age_categories[temp]

    gender = 'male' if gender_pred[0][0] > gender_pred[0][1] else 'female'

    return age, gender

@app.route('/', methods=['GET'])
def index():
    # Main page
    return 'Gender and Age Prediction!'

@app.route('/api/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    f = request.files['file']

    if f.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    try:
        age, gender = model_predict(file_path)
        os.remove(file_path)  # Remove the file after prediction
        return jsonify({'age': age, 'gender': gender})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

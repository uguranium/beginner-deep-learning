from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

cwd = os.getcwd()
model_path = cwd+'/models/vgg19.h5'

if os.path.isfile(model_path):
    model = load_model(model_path)
else:
    model = VGG19()
    model.save(model_path)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model.predict(x)


app = Flask(__name__, template_folder='html_templates')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(cwd, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        prediction = model_predict(file_path, model)
        decode_prediction = decode_predictions(prediction, top=1)
        print(decode_prediction)
        return str(decode_prediction)

if __name__ == '__main__':
    app.run(debug=True)
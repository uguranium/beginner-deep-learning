from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template

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


app = Flask(__name__)


@app.route('/', method=['GET'])
def index():
    return render_template(cwd+'/index.html')


if __name__ == '__main__':
    app.run(dabug=True)
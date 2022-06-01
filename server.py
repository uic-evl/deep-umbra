import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import traceback
import json
import numpy as np
from utils import *
from tiled_elevation import * 
from deep_shadow import *
from flask import Flask, request
from flask import jsonify

IMG_PATH = os.getenv('APP_ROOT', '/infer/img')
JSON_PATH = os.getenv('APP_ROOT', '/infer/json')

app = Flask(__name__)

# model = tf.keras.models.load_model('savedmodels/1')
deepShadow = DeepShadow(512,512)
deepShadow.restore('checkpoints/cktp-8')
model = deepShadow.generator

def load_grid_from_osm(date, zoom, xtile, ytile, max_height):
    all_input = tf.zeros((256*3,256*3,1))

    for x in range(-1,2):
        for y in range(-1,2):
            try: # Tiles with no buildings raise error. They should be ignored
                iinput = get_image(xtile+y, ytile+x, zoom, max_height) # Gets image from osm
                iinput = tf.reshape(tf.convert_to_tensor(iinput), (256, 256, 1))
                iinput = tf.cast(iinput, tf.float32)
                indices = [(xx,yy) for xx in range(256+256*x,256+256*(x+1)) for yy in range(256+256*y,256+256*(y+1))]
                indices = np.array(indices).reshape(256,256,-1)
                all_input = tf.tensor_scatter_nd_update(all_input, indices, iinput)
            except KeyError:
                continue
            
    (latitude, longitude) = num2deg(xtile, ytile, zoom)
            
    all_input = all_input[128:-128,128:-128]
    all_lat = tf.ones((512,512), dtype=tf.float32)
    all_lat = tf.math.scalar_mul(float(latitude), all_lat)
    all_lat = tf.reshape(all_lat, (512, 512, 1))
    
    if date == 'winter':
        value = 0
    elif date == 'spring':
        value = 1
    else:
        value = 2
        
    all_date = tf.ones((512,512), dtype=tf.float32)
    all_date = tf.math.scalar_mul(float(value), all_date)
    all_date = tf.reshape(all_date, (512, 512, 1))
        
    return all_input, all_lat, all_date

@app.route(IMG_PATH, methods=["POST"])
def inferImg():
    data = request.json
    instance = data['instance']
    example_input = tf.convert_to_tensor(instance[0])
    example_lat = tf.convert_to_tensor(instance[1])
    example_dat = tf.convert_to_tensor(instance[2])
    clamped_prediction = np.squeeze(model([example_input, example_lat, example_dat], training=True).numpy()[:,128:-128,128:-128,:])
    return json.dumps(clamped_prediction.tolist())
    
@app.route(JSON_PATH, methods=["POST"])
def inferJson():
    data = request.json
    info = data['info']
    xtile = info[0]
    ytile = info[1]
    zoom = info[2]
    max_height = info[3]
    season = info[4]
    example_input, example_lat, example_dat = load_grid_from_osm(season, zoom, xtile, ytile, max_height)
    example_input, example_lat, example_dat = normalize_input(example_input, example_lat, example_dat)
    example_input = np.array(example_input).reshape(1,512,512,1)
    example_lat = np.array(example_lat).reshape(1,512,512,1)
    example_dat = np.array(example_dat).reshape(1,512,512,1)
    clamped_prediction = np.squeeze(model([example_input, example_lat, example_dat], training=True).numpy()[:,128:-128,128:-128,:])
    return json.dumps(clamped_prediction.tolist())

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())
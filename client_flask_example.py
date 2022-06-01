
import requests
import matplotlib.pyplot as plt
import numpy as np

from utils import *

def request_using_image(height_path, city, date, zoom, i, j):
    input_height, input_lat, input_date = load_input_grid(height_path, city, date, zoom, i, j)
    input_height, input_lat, input_date = normalize_input(input_height, input_lat, input_date)
    input_height = np.array(input_height).reshape(1,512,512,1)
    input_lat = np.array(input_lat).reshape(1,512,512,1)
    input_date = np.array(input_date).reshape(1,512,512,1)

    data = { 'instance': [input_height.tolist(), input_lat.tolist(), input_date.tolist()] }
    response = requests.post('http://0.0.0.0:5000/infer/img', json = data)
    response.raise_for_status()
    prediction = response.json()

    plt.figure(figsize=(15, 15))
    plt.imshow(np.array(prediction) * 0.5 + 0.5)
    plt.axis('off')
    plt.savefig("prediction.png")

def request_using_json(xtile, ytile, zoom, maxheight, season):
    data = { 'info': [xtile, ytile, zoom, maxheight, season] }
    response = requests.post('http://0.0.0.0:5000/infer/json', json = data)
    response.raise_for_status()
    image = response.json()
    plt.figure(figsize=(15, 15))
    plt.imshow(np.array(image) * 0.5 + 0.5)
    plt.axis('off')
    plt.savefig("prediction.png")

request_using_json(42661, 28171, 16, 550, 'summer')
# request_using_image('data/heights', 'nyc', 'summer', 16, 19289, 24591)








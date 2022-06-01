# deep-shadows


## How to run the server

- Install flask with pip
- Run `$ export FLASK_APP=server`
- Run `$ flask run`  

## Making requests to the server

- A example of client can be seen in "client_flask_example.py"

### Routes

- `http://0.0.0.0:5000/infer/img`:
    - `POST`: Returns a python list (values are in the interval [-1,1]) representing the predicted shadow scheme given an input image
        - JSON format: { 'instance': [input_image : list, input_lat : list, input_date : list]}
            - input_image: a python list of shape (1,512,512,1) containing the height description of a region. It must be normalized (values in the interval [-1,1]).
            - input_lat: a python list of shape (1,512,512,1) with all positions containing the same value: the latitude of the region described in input_image. It must be normalized (values in the interval [-1,1]).
            - input_date: a python list of shape (1,512,512,1) with all positions containing the same value (represents season): 0 = winter, 1 = spring and 3 = summer. It must be normalized (values in the interval [-1,1]).
- `http://0.0.0.0:5000/infer/json`:
    - `POST`: Returns a python list (values are in the interval [-1,1]) representing the predicted shadow scheme given an input json describing a slippy tile
        - JSON format: { 'info': [xtile : int, ytile : int, zoom : int, maxheight : float | int, season : string] }
            - xtile: x coordinate of the slippy tile
            - ytile: y coordinate of the slippy tile
            - zoom: zoom level of the slippy tile
            - maxheight: max height present in the dataset
            - season: string representing the season ('summer', 'winter', 'spring')
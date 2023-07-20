# Faulty Image API

This API combines the image classification models (i.e. empty, sticky notes, folded corner and writing type classifiers) trained in Dalai project. Using this API one can predict whether the document is empty, contains sticky notes or folded corners and the writing type of the document. This API is also used in [arkkiivi](https://arkkiivi.fi/) backend. 

## Installation

- Create and activate conda environment:

`conda create -n faulty_api_env python=3.7`

`conda activate faulty_api_env`

- Install poppler:

`conda install -c conda-forge poppler`

- Install required libraries:

`pip install -r requirements.txt`

## How to use

- Open flask app: 

`flask --app api.py run`

- Open flask app with debug: 

`flask --app api.py --debug run`

The API assumes that the models are found in './mallit' folder. The models are named 'post_it_model_20122022.onnx', 'corner_model_19122022.onnx', 'empty_model_v4.onnx' ja 'writing_type_v1.onnx'.

The trained models are transformed into the [ONNX](https://onnx.ai/) format in order to speed up inference and to make the use of the model less dependent on specific frameworks and libraries. 

- The API has one endpoint called `/detect`. The desired models used in inference can be chosen with arguments in the POST http request. Argument 0 means that the model is not used and 1 means that it is used.

- An example of POST http request where all models are used: `/detect?postit=1&corner=1&empty=1&writing_type=1`

- Default-port: 5000

- The API expects that the image file of the document is attached to the POST request. One can test the API with request.py file or with for example curl command:

`curl http://127.0.0.1:5000/detect?postit=1&corner=1&empty=1&writing_type=1 -F image=@/path/img.jpg` 

The API returns a Flask response that contains a csv file called virheet.csv.

import os
import PIL
import onnxruntime
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from pdf2image import convert_from_bytes
from flask import Flask, request
import io
from flask_cors import CORS


# Creates the Flask app
app = Flask(__name__)
CORS(app)

# Transform methods for corner & post-it model inputs
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

# Transform methods for empty model inputs
empty_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

writing_type_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.882, 0.883, 0.899], [0.088, 0.089, 0.094])
    ])

# Models are expected to be saved in ./mallit folder in the same path as the code
MODEL_PATH = './mallit/' 
POST_IT_MODEL = 'post_it_model_20122022.onnx'
CORNER_MODEL = 'corner_model_19122022.onnx'
EMPTY_MODEL = 'empty_model_v4.onnx'
WRITING_TYPE_MODEL = 'writing_type_v1.onnx'

try:
    # Load the models and the trained weights
    post_it_model = onnxruntime.InferenceSession(MODEL_PATH + POST_IT_MODEL)
    corner_model = onnxruntime.InferenceSession(MODEL_PATH + CORNER_MODEL)
    empty_model = onnxruntime.InferenceSession(MODEL_PATH + EMPTY_MODEL)
    writing_type_model = onnxruntime.InferenceSession(MODEL_PATH + WRITING_TYPE_MODEL)
except Exception as e:
    print("Failed to load pretrained models: {}".format(e))

# Method for getting prediction results for the models
def predict_fault(image, model):
    image = image.detach().cpu().numpy()
    input = {model.get_inputs()[0].name: image}
    output = model.run(None, input)
    preds = np.argmax(output[0], 1)
    return preds.item()

# Method for getting predictions based on the arguments given in the query string
def predict_labels(images, args, name, extension, col_names,idj):
    result_list = []
    # Loop over images
    for i, image in enumerate(images):
        res_dict = {key: None for key in col_names}
        page = ''
        if len(images) > 1:
            # Add page number to each image extracted from multi-page pdf
            page = '_page_' + str(i + 1)

        # Add image name to result dict
        res_dict['file'] = name + page + extension

        input_image = None 

        # Post-it prediction
        if args[0] == '1':
            input_image = data_transforms(image).unsqueeze(0)
            label = predict_fault(input_image, post_it_model)
            res_dict['post_it'] = label

        # Folded corner prediction 
        if args[1] == '1':
            # Perform transformations only when needed
            input_image = data_transforms(image).unsqueeze(0) if input_image is None else input_image
            label = predict_fault(input_image, corner_model)
            res_dict['corner'] = label

        # Empty document prediction
        if args[2] == '1':
            # Transformations for empty page detection
            input_image = empty_transforms(image).unsqueeze(0)
            label = predict_fault(input_image, empty_model)
            res_dict['empty'] = 1 - label  

        if args[3] == '1':
            # Transformations for writing_type page detection
            input_image = writing_type_transforms(image).unsqueeze(0)
            label = predict_fault(input_image, writing_type_model)
            res_dict['writing_type'] = label
            
        res_dict['id']= idj     

        # Result is a list of dicts, one for each processed page
        result_list.append(res_dict)

    return result_list


@app.route('/detect', methods=["POST"])
def detect():
    # Extract arguments from query string
    postit  = request.args.get('postit', None)
    corner  = request.args.get('corner', None)
    empty  = request.args.get('empty', None)
    writing_type  = request.args.get('writing_type', None)
    idj = request.args.get('id',None)
    # Save arguments in a list
    args = [postit, corner, empty, writing_type, idj]

    # Expects message body to have the key "image"
    if request.files.get("image"):
        try:
            image_file = request.files["image"]
        except Exception as e:
            print("Failed to load input image: {}".format(e))

        # Read the file in bytes form
        image_bytes = image_file.read()
        # Get the name of the image/pdf file
        file_name = image_file.filename
        # Split file name to body and extension (.pdf, .jpg etc.)
        name, extension = os.path.splitext(file_name)
       
        if extension == '.pdf':
            # Convert pdf file to a list of image files (one per each document page)
            images = convert_from_bytes(image_bytes)
        else:
            image = PIL.Image.open(io.BytesIO(image_bytes))
            image.draft('RGB', (224, 224))
            images = [image.convert("RGB")]

        # Column names for the .csv file
        col_names = ['file','post_it','corner','empty', 'writing_type', 'id']
        # Get prediction results for each image as a list of dicts
        result_list = predict_labels(images, args, name, extension, col_names,idj)
        return result_list
    else:
        print("POST-request does not contain input image.")

if __name__ == "__main__":
    #app.run(port=, host='')

    app.run(port=5000)

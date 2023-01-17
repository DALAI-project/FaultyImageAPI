import pprint
from PIL import Image
import json

import requests

DETECTION_URL = "http://localhost:5000/detect?postit=0&corner=0&empty=0"
FILE_PATH = "./data/"
FILE_NAME = "postit_001.jpg"

response = requests.post(DETECTION_URL, files={'image': open(FILE_PATH + FILE_NAME, 'rb')})

print('.csv content: ', response.content)
print('Headers: ', response.headers)

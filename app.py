from flask import Flask, request, jsonify
from werkzeug.serving import WSGIRequestHandler
import base64
import cv2
from PIL import Image
import io
import numpy as np
from main import img_text

app = Flask(__name__)


@app.route('/v1/image/convert_text', methods=['POST'])
def convert_text():
    data = request.get_json()

    if 'image' not in data:
        return "", 400

    else:
        captured_img_data = convert(data['image'])
        returned_text = img_text(captured_img_data)
        return returned_text, 200


@app.route("/")
def index():
    return "<h1>Welcome to lsh ml server !!</h1>"

# ref. https://ballentain.tistory.com/50

# Binary 형태로 이미지 데이터 읽은 다음 decode 하는 방법
def stringToRGB(imgdata):
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def convert(image):
    imgdata_bytes = base64.b64decode(image)
    image = stringToRGB(imgdata_bytes)
    return image


if __name__ == "__main__":
    # https://stackoverflow.com/questions/63765727/unhandled-exception-connection-closed-while-receiving-data
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(threaded=True, host='0.0.0.0', port=5000)

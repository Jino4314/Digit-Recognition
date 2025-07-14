from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import re

app = Flask(__name__)
model = load_model('mnist_digit_recognition.h5')
submitted_digit = None

def preprocess_base64_image(image_data):
    image_str = re.search(r'base64,(.*)', image_data).group(1)
    img_bytes = base64.b64decode(image_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img = preprocess_base64_image(data['image'])
    pred = model.predict(img)
    digit = int(np.argmax(pred))
    return jsonify({'digit': digit})

@app.route('/submit_digit', methods=['POST'])
def submit_digit():
    global submitted_digit
    data = request.get_json()
    submitted_digit = data['digit']
    return jsonify({'message': f'Digit {submitted_digit} submitted!'})

@app.route('/guess_digit', methods=['POST'])
def guess_digit():
    data = request.get_json()
    guess = data['guess']
    if submitted_digit is None:
        return jsonify({'result': 'No digit submitted yet.'})
    result = 'Correct!' if str(guess) == str(submitted_digit) else f'Wrong! It was {submitted_digit}'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)


from io import BytesIO
import numpy as np
import requests
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ultralytics import YOLO
import base64

model = YOLO(f'best.pt')

app = Flask(
    __name__,
    static_folder='public',
    static_url_path='/'
)

CORS(app, resources={r"/*": {"origins": "*"}})



origins = [
    "https://vercel-opencv.vercel.app/",
    "http://localhost:8080",
]


@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    # Get image from request (assuming image is sent as multipart/form-data)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    image = file.read()

    
    # Load the image using opencv
    file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Create the mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # Create a polygon from points for the mask
    prediction = model.predict(img)
    pts = np.array([[x, y] for (x, y) in zip(prediction[0].summary()[0]['segments']['x'], prediction[0].summary()[0]['segments']['y'])], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # Slice the image using the mask
    sliced_img = cv2.bitwise_and(img, img, mask=mask)
    npoints_crown = (mask > 0).sum()
    
    # Convert image to HSV
    hsv_img = cv2.cvtColor(sliced_img, cv2.COLOR_RGB2HSV)
    
    # Create hue mask for values between 32 and 85
    lower_hue = np.array([35, 10, 10])
    upper_hue = np.array([90, 255, 255])
    hue_mask = cv2.inRange(hsv_img, lower_hue, upper_hue)

    # Calculate percentage of pixels matching the hue mask
    npoints_leaf = (hue_mask > 0).sum()
    percentage = npoints_leaf/npoints_crown * 100

    # Sliced image to JPEG format
    _, buffer = cv2.imencode('.jpg', sliced_img)
    # Convert the buffer to a Base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    # Format the Base64 string to be used in HTML <img> tag
    crown_base64 = f"data:image/jpeg;base64,{base64_string}"
    
    # HSV masked image to JPEG format
    hsv_masked_img = cv2.bitwise_and(sliced_img, sliced_img, mask=hue_mask)
    _, buffer = cv2.imencode('.jpg', hsv_masked_img)
    # Convert the buffer to a Base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    # Format the Base64 string to be used in HTML <img> tag
    leaf_base64 = f"data:image/jpeg;base64,{base64_string}"

    # Return the percentage as JSON
    return jsonify({"matching_percentage": percentage, "crown_image": crown_base64, "leaf_image": leaf_base64})


if __name__ == "__main__":
    app.run(debug=True)
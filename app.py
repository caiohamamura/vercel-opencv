from io import BytesIO
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

origins = [
    "https://vercel-opencv.vercel.app/",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/", StaticFiles(directory="public", html = True), name="public")


# Roboflow configuration
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_URL = "https://detect.roboflow.com/fenologia-tcc/3"

@app.post("/analyze-image")
async def analyze_image(request: Request):
    # Get image from request (assuming image is sent as multipart/form-data)
    form = await request.form()
    image = form["file"].file.read()

    # Send image to Roboflow API
    response = requests.post(
        ROBOFLOW_MODEL_URL,
        files={"file": image},
        params={"api_key": ROBOFLOW_API_KEY}
    )

    # Parse response
    response_json = response.json()
    if "predictions" not in response_json or len(response_json["predictions"]) == 0:
        return {"error": "No predictions found"}

    # Extract the first prediction
    prediction = response_json["predictions"][0]

    # Load the image using opencv
    image.seek(0)
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Create the mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # Create a polygon from points for the mask
    points = prediction["points"]
    pts = np.array([[point["x"], point["y"]] for point in points], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # Slice the image using the mask
    sliced_img = cv2.bitwise_and(img, img, mask=mask)
    npoints_crown = (mask > 0).sum()
    
    # Convert image to HSV
    hsv_img = cv2.cvtColor(sliced_img, cv2.COLOR_RGB2HSV)
    
    # Create hue mask for values between 32 and 85
    lower_hue = np.array([32, 10, 200])
    upper_hue = np.array([85, 255, 255])
    hue_mask = cv2.inRange(hsv_img, lower_hue, upper_hue)

    # Calculate percentage of pixels matching the hue mask
    npoints_leaf = (hue_mask > 0).sum()
    npoints_leaf/npoints_crown * 100
    matching_pixels = np.sum(hue_mask)
    percentage = (matching_pixels / total_pixels) * 100

    

    # Return the percentage as JSON
    return {"matching_percentage": percentage}

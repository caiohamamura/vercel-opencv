import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/hsv_mask', methods=['POST'])
def hsv_mask():
    try:
        # Get the image from the request
        file = request.files['image']
        image_bytes = file.read()
        
        # Convert the image to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define Hue range (32 to 85)
        lower_bound = np.array([32, 50, 50])
        upper_bound = np.array([85, 255, 255])

        # Create a mask based on the hue range
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Calculate percentage of image matching the mask
        mask_percentage = (np.sum(mask > 0) / mask.size) * 100

        # Return the result as JSON
        return jsonify({
            'mask_percentage': mask_percentage
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run()

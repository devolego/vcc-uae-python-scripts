from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import imutils
import os
from pyimagesearch.transform import four_point_transform
from PIL import Image, ImageOps

app = Flask(__name__)

@app.route('/scan', methods=['POST'])
def scan_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Save the file temporarily
    image_path = os.path.join('temp', file.filename)
    file.save(image_path)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Compute the ratio of the old height to the new height, clone it,
    # and resize it easier for compute and viewing
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    
    # Convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blurring to remove high frequency noise helping in
    # Contour Detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection
    edged = cv2.Canny(gray, 75, 200)
    
    # Finding the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Handling due to different versions of OpenCV
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Taking only the top 5 contours by Area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    
    # Looping over the contours
    for c in cnts:
        # Approximating the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 0.02
        
        # If our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # After the loop
    if screenCnt is None or not hasattr(screenCnt, '__len__') or len(screenCnt) == 0:
        cv2.imwrite('temp/warped.jpg', image)
        return send_file('temp/warped.jpg', mimetype='image/jpeg')
    
    # Draw the contour (outline)
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imwrite("temp/boundary_detected.jpg", image)
    
    # Apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
    # Save the warped image
    warped_path = "temp/warped.jpg"
    cv2.imwrite(warped_path, warped)
    
    # Return the processed image
    return send_file(warped_path, mimetype='image/jpeg')


def preprocess_image(input_path, output_path):
    # Open the image file
    image = Image.open(input_path)
    
    # Resize the image to a smaller width while maintaining aspect ratio
    basewidth = 1000
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((basewidth, hsize), Image.LANCZOS)
    
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # Apply thresholding
    threshold = 100
    image = image.point(lambda p: p > threshold and 255)
    
    # Save the preprocessed image
    image.save(output_path)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Save the file temporarily
    input_path = os.path.join('temp', file.filename)
    file.save(input_path)
    
    output_path = os.path.join('temp', 'preprocessed_python.jpg')
    
    # Preprocess the image
    preprocess_image(input_path, output_path)
    
    # Return the processed image
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
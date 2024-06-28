from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import io
from pyimagesearch.transform import four_point_transform
import cv2
import imutils

app = Flask(__name__)

def generate_large_image(size_in_mb, font_size):
    # Start with a large canvas
    width, height = 5000, 5000
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # Load the specified font size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Estimate number of iterations required
    avg_char_size = font_size * 0.6  # approximate width of a character
    text_length = len("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10)
    chars_per_line = width // avg_char_size
    lines_per_iteration = height // (font_size + 10)
    chars_per_iteration = chars_per_line * lines_per_iteration
    estimated_iterations = (size_in_mb * 1024 * 1024) // (chars_per_iteration * avg_char_size)
    
    # Add text until the file size reaches the desired size
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
    y_text = 10
    with tqdm(total=size_in_mb * 1024 * 1024, unit='B', unit_scale=True) as pbar:
        while True:
            for y in range(10, height, font_size + 10):
                draw.text((10, y), text, font=font, fill=(0, 0, 0))
                pbar.update(len(text) * avg_char_size)
            buffer = io.BytesIO()
            image.save(buffer, 'JPEG')
            buffer.seek(0)
            if buffer.getbuffer().nbytes >= size_in_mb * 1024 * 1024:
                break

    return buffer

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        size_in_mb = data.get('size_in_mb', 4)
        font_size = data.get('font_size', 50)

        image_buffer = generate_large_image(size_in_mb, font_size)

        return send_file(image_buffer, mimetype='image/jpeg', as_attachment=True, attachment_filename='large_image.jpg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Compute the ratio of the old height to the new height, clone it,
    # and resize it easier for compute and viewing
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # Convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blurring to remove high frequency noise helping in Contour Detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection
    edged = cv2.Canny(gray, 75, 200)

    # Finding the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Taking only the top 5 contours by Area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None

    # Looping over the contours
    for c in cnts:
        # Approximating the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # If our approximated contour has four points, then we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None or not hasattr(screenCnt, '__len__') or len(screenCnt) == 0:
        return image_path

    # Apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # Save the warped image
    output_path = os.path.join('temp', 'warped.jpg')
    cv2.imwrite(output_path, warped)

    return output_path

@app.route('/scan', methods=['POST'])
def scan_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        image_path = os.path.join('temp', image_file.filename)
        image_file.save(image_path)

        processed_image_path = process_image(image_path)

        return send_file(processed_image_path, mimetype='image/jpeg', as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

from flask import Flask, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import io
from pyimagesearch.transform import four_point_transform
import cv2
import imutils
from qreader import QReader

app = Flask(__name__)

# def generate_large_image(size_in_mb, font_size):
#     # Start with a large canvas
#     width, height = 5000, 5000
#     image = Image.new('RGB', (width, height), color='white')
#     draw = ImageDraw.Draw(image)

#     # Load the specified font size
#     try:
#         font = ImageFont.truetype("arial.ttf", font_size)
#     except IOError:
#         font = ImageFont.load_default()

#     # Estimate number of iterations required
#     avg_char_size = font_size * 0.6  # approximate width of a character
#     text_length = len("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10)
#     chars_per_line = width // avg_char_size
#     lines_per_iteration = height // (font_size + 10)
#     chars_per_iteration = chars_per_line * lines_per_iteration
#     estimated_iterations = (size_in_mb * 1024 * 1024) // (chars_per_iteration * avg_char_size)
    
#     # Add text until the file size reaches the desired size
#     text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
#     y_text = 10
#     with tqdm(total=size_in_mb * 1024 * 1024, unit='B', unit_scale=True) as pbar:
#         while True:
#             for y in range(10, height, font_size + 10):
#                 draw.text((10, y), text, font=font, fill=(0, 0, 0))
#                 pbar.update(len(text) * avg_char_size)
#             buffer = io.BytesIO()
#             image.save(buffer, 'JPEG')
#             buffer.seek(0)
#             if buffer.getbuffer().nbytes >= size_in_mb * 1024 * 1024:
#                 break

#     return buffer

# @app.route('/generate_image', methods=['POST'])
# def generate_image():
    # try:
    #     data = request.get_json()
    #     size_in_mb = data.get('size_in_mb', 4)
    #     font_size = data.get('font_size', 50)

    #     image_buffer = generate_large_image(size_in_mb, font_size)

    #     return send_file(image_buffer, mimetype='image/jpeg', as_attachment=True, attachment_filename='large_image.jpg')
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

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


def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": f"Failed to fetch the website content. Status code: {response.status_code}"}

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the required information
    vehicle_details = {}

    def get_detail(label):
        element = soup.find(lambda tag: tag.name == "label" and tag.string and tag.string.strip() == label.strip())
        if element:
            sibling = element.find_next_sibling('div')
            if sibling and sibling.find('span'):
                return sibling.find('span').text.strip()
        return None

    vehicle_details['VCC No'] = get_detail('VCC No :')
    vehicle_details['Chassis No'] = get_detail('Chassis No :')
    vehicle_details['Year of Built'] = get_detail('Year of Built :')
    vehicle_details['Country of Origin'] = get_detail('Country of Origin :')
    vehicle_details['Vehicle Model'] = get_detail('Vehicle Model :')
    vehicle_details['Vehicle Brand Name'] = get_detail('Vehicle Brand Name :')
    vehicle_details['Vehicle Type'] = get_detail('Vehicle Type :')
    vehicle_details['Vehicle Color'] = get_detail('Vehicle Color :')
    vehicle_details['Declaration Number'] = get_detail('Declaration Number :')
    vehicle_details['Declaration Date'] = get_detail('Declaration Date :')
    vehicle_details['Owner Name'] = get_detail('Owner Name :')

    return vehicle_details

@app.route('/scrape', methods=['POST'])
def scrape():
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        scraped_data = scrape_website(url)
        return jsonify(scraped_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def read_qr_code(image_path):
    # Create a QReader instance
    qreader = QReader()

    # Get the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Use the detect and decode function
    decoded_text = qreader.detect_and_decode(image=image)

    return decoded_text

@app.route('/read_qr', methods=['POST'])
def read_qr():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        image_path = os.path.join('temp', image_file.filename)
        image_file.save(image_path)

        qr_code_text = read_qr_code(image_path)

        return jsonify({"decoded_text": qr_code_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

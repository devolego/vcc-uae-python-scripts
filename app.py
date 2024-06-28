from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import io

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

import os
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.feature import hog
import uuid 

app = Flask(__name__)

# Load the trained model
model = load_model('recognition_model_canny_edge.keras')

# Define a mapping of class indices to Devanagari characters and phonetics
class_mapping = {
    0: {'devanagari': 'क', 'phonetics': 'ka'},
    1: {'devanagari': 'ख', 'phonetics': 'kha'},
    2: {'devanagari': 'ग', 'phonetics': 'ga'},
    3: {'devanagari': 'घ', 'phonetics': 'gha'},
    4: {'devanagari': 'ङ', 'phonetics': 'ṅa'},
    5: {'devanagari': 'च', 'phonetics': 'ca'},
    6: {'devanagari': 'छ', 'phonetics': 'cha'},
    7: {'devanagari': 'ज', 'phonetics': 'ja'},
    8: {'devanagari': 'झ', 'phonetics': 'jha'},
    9: {'devanagari': 'ञ', 'phonetics': 'ña'},
    10: {'devanagari': 'ट', 'phonetics': 'ṭa'},
    11: {'devanagari': 'ठ', 'phonetics': 'ṭha'},
    12: {'devanagari': 'ड', 'phonetics': 'ḍa'},
    13: {'devanagari': 'ढ', 'phonetics': 'ḍha'},
    14: {'devanagari': 'ण', 'phonetics': 'ṇa'},
    15: {'devanagari': 'त', 'phonetics': 'ta'},
    16: {'devanagari': 'थ', 'phonetics': 'tha'},
    17: {'devanagari': 'द', 'phonetics': 'da'},
    18: {'devanagari': 'ध', 'phonetics': 'dha'},
    19: {'devanagari': 'न', 'phonetics': 'na'},
    20: {'devanagari': 'प', 'phonetics': 'pa'},
    21: {'devanagari': 'फ', 'phonetics': 'pha'},
    22: {'devanagari': 'ब', 'phonetics': 'ba'},
    23: {'devanagari': 'भ', 'phonetics': 'bha'},
    24: {'devanagari': 'म', 'phonetics': 'ma'},
    25: {'devanagari': 'य', 'phonetics': 'ya'},
    26: {'devanagari': 'र', 'phonetics': 'ra'},
    27: {'devanagari': 'ल', 'phonetics': 'la'},
    28: {'devanagari': 'व', 'phonetics': 'va'},
    29: {'devanagari': 'श', 'phonetics': 'śa'},
    30: {'devanagari': 'ष', 'phonetics': 'ṣa'},
    31: {'devanagari': 'स', 'phonetics': 'sa'},
    32: {'devanagari': 'ह', 'phonetics': 'ha'},
    33: {'devanagari': 'क्ष', 'phonetics': 'kṣa'},
    34: {'devanagari': 'त्र', 'phonetics': 'tra'},
    35: {'devanagari': 'ज्ञ', 'phonetics': 'jña'},
    36: {'devanagari': '०', 'phonetics': '०'},
    37: {'devanagari': '१', 'phonetics': '१'},
    38: {'devanagari': '२', 'phonetics': '२'},
    39: {'devanagari': '३', 'phonetics': '३'},
    40: {'devanagari': '४', 'phonetics': '४'},
    41: {'devanagari': '५', 'phonetics': '५'},
    42: {'devanagari': '६', 'phonetics': '६'},
    43: {'devanagari': '७', 'phonetics': '७'},
    44: {'devanagari': '८', 'phonetics': '८'},
    45: {'devanagari': '९', 'phonetics': '९'},
    46: {'devanagari': '्/अ', 'phonetics': 'a'},
    47: {'devanagari': 'ा/आ', 'phonetics': 'ā'},
    48: {'devanagari': 'ि/इ', 'phonetics': 'i'},
    49: {'devanagari': 'ी/ई', 'phonetics': 'ī'},
    50: {'devanagari': 'ु/उ', 'phonetics': 'u'},
    51: {'devanagari': 'ू/ऊ', 'phonetics': 'ū'},
    52: {'devanagari': 'े/ए', 'phonetics': 'e'},
    53: {'devanagari': 'ै/ऐ', 'phonetics': 'ai'},
    54: {'devanagari': 'ो/ओ', 'phonetics': 'o'},
    55: {'devanagari': 'ौ/औ', 'phonetics': 'au'},
    56: {'devanagari': 'ं/अं', 'phonetics': 'aṁ'},
    57: {'devanagari': 'ः/अः', 'phonetics': 'aḥ'},
    58: {'devanagari': 'ॐ', 'phonetics': 'om'},
    59: {'devanagari': 'ँ/अँ', 'phonetics': 'anunāsika'},
    60: {'devanagari': 'ॠ ', 'phonetics': 'ri'},
    61: {'devanagari': '।', 'phonetics': 'pūrna virāma'},
    62: {'devanagari': '॥', 'phonetics': 'deerga virāma'},
}

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    resized_image = cv2.resize(image, (28, 28))
    # Apply edge detection
    edges = cv2.Canny(resized_image, 100, 200)
    # Extract HOG features
    hog_features = hog(edges, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return np.array([hog_features])

# Define a route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling image upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the file in binary mode and convert it to a NumPy array
        image_data = np.frombuffer(file.read(), np.uint8)
        # Decode the image data into a grayscale image
        image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        
        # Generate a unique filename for each upload using UUID
        temp_filename = 'static/uploaded_image_' + str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(temp_filename, image)
                
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction))

        # Get the Devanagari label and phonetics for the predicted class from the mapping
        if predicted_class in class_mapping:
            devanagari_label = class_mapping[predicted_class]['devanagari']
            phonetics = class_mapping[predicted_class]['phonetics']
        else:
            devanagari_label = 'Unknown'
            phonetics = 'Unknown'

        return jsonify({'uploaded_image': temp_filename, 'devanagari_label': devanagari_label, 'phonetics': phonetics})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

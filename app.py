import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, Response, jsonify, request, jsonify
import time
import spacy
import os

app = Flask(__name__)

# Initialize Mediapipe Holistic Model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Load the trained CNN model
model = tf.keras.models.load_model('sign_language_cnn_model_word11150.h5')

# Load the LabelEncoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes11150.npy')

# Define image size (same as during model training)
IMG_SIZE = 64

# Initialize camera and other variables
cap = None
current_prediction = 'None'
sentence = ""
last_sentence = ""
current_word = ""
start_time = None
last_detection_time = None
SENTENCE_TIMEOUT = 0.75  # seconds to consider a word continuous
INACTIVITY_TIMEOUT = 2.0  # seconds to wait before updating last sentence when no hand is detected

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

def predict_sign(image):
    global current_prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    current_prediction = label_encoder.inverse_transform([predicted_class])[0]
    return current_prediction

def put_transparent_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=1, alpha=0.0): #text visibility increase alpha
    overlay = frame.copy()
    
    # Create an overlay with an alpha channel
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    
    # Text size and position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    text_y += text_size[1]
    
    # Put text on the overlay with alpha channel
    cv2.putText(overlay, text, (text_x + 1, text_y + 1), font, font_scale, (255, 255, 255, 255), thickness + 2, cv2.LINE_AA)  # White text with no transparency
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, color + (int(alpha * 255),), thickness, cv2.LINE_AA)  # Colored text with transparency
    
    # Convert back to BGR and blend with original frame
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame


def generate_frames():
    global cap, current_prediction, sentence, last_sentence, current_word, start_time, last_detection_time, SENTENCE_TIMEOUT, INACTIVITY_TIMEOUT
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        current_time = time.time()
        detected_hand = results.left_hand_landmarks or results.right_hand_landmarks

        if detected_hand:
            last_detection_time = current_time  # Update last detection time when hand is detected
            current_prediction = predict_sign(frame)
            
            if current_prediction != "Sign Not Found":
                if current_prediction == current_word:
                    if start_time and (current_time - start_time) >= SENTENCE_TIMEOUT:
                        if len(sentence) == 0 or sentence.split()[-1] != current_prediction:
                            sentence += f"{current_prediction} "
                        current_word = None
                else:
                    current_word = current_prediction
                    start_time = current_time
            else:
                current_word = ""
                start_time = None

        else:
            if sentence:
                if last_detection_time and (current_time - last_detection_time) >= INACTIVITY_TIMEOUT:
                    last_sentence = sentence.strip()
                    sentence = ""
                    current_word = ""
                    start_time = None
                    last_detection_time = None

        frame = put_transparent_text(frame, f"Sign: {current_prediction}", (10, 30))
        frame = put_transparent_text(frame, f"Sentence: {sentence.strip()}", (10, 60))
        frame = put_transparent_text(frame, f"Last Sentence: {last_sentence}", (10, 90))
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        # Yield the frame as bytes
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def current_prediction_route():
    return jsonify({'prediction': current_prediction, 'sentence': sentence, 'last_sentence': last_sentence})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return jsonify({'message': 'Camera restarted'})
    else:
        return jsonify({'message': 'Cannot open camera'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap and cap.isOpened():
        cap.release()
        cap = None
        return jsonify({'message': 'Camera stopped'})
    return jsonify({'message': 'Camera is not running'})



#   text to aniamation  genrator code 


@app.route('/animation', methods=['GET', 'POST'])
def animation_view():
    if request.method == 'POST':
        text = request.form.get('sen', '').lower()

        # Process the text using spaCy
        doc = nlp(text)

        # Initialize tense counters
        tense = {
            "future": 0,
            "present": 0,
            "past": 0,
            "present_continuous": 0
        }

        filtered_text = []
        for token in doc:
            pos = token.pos_
            tag = token.tag_

            # Count tenses
            if tag == "MD":
                tense["future"] += 1
            elif pos in ["VERB", "AUX"]:
                if tag in ["VBG", "VBN", "VBD"]:
                    tense["past"] += 1
                elif tag == "VBG":
                    tense["present_continuous"] += 1
                else:
                    tense["present"] += 1

            # Lemmatization
            if pos in ["VERB", "NOUN"]:
                filtered_text.append(token.lemma_)
            elif pos in ["ADJ", "ADV"]:
                filtered_text.append(token.lemma_)
            else:
                filtered_text.append(token.text)

        probable_tense = max(tense, key=tense.get)

        if probable_tense == "past" and tense["past"] >= 1:
            filtered_text = ["Before"] + filtered_text
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in filtered_text:
                filtered_text = ["Will"] + filtered_text
        elif probable_tense == "present":
            if tense["present_continuous"] >= 1:
                filtered_text = ["Now"] + filtered_text

        # Handle static files
        processed_words = []
        for w in filtered_text:
            path = os.path.join(app.static_folder, 'words', f'{w}.mp4')
            if not os.path.exists(path):
                processed_words.extend(list(w))
            else:
                processed_words.append(w)
        filtered_text = processed_words

        return render_template('index.html', words=filtered_text, text=text)
    else:
        return render_template('index.html')





if __name__ == '__main__':
    app.run(debug=True)

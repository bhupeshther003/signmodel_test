from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import time
import spacy
import os
import threading

app = Flask(__name__)

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Initialize Mediapipe Holistic Model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load the trained CNN model
model = tf.keras.models.load_model('sign_language_cnn_model_word11150.h5')

# Load the LabelEncoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes11150.npy')

# Define image size (same as during model training)
IMG_SIZE = 64

# Initialize camera and other variables
cap = None
running = False
sentence = ""
last_sentence = ""
current_word = ""
start_time = None
SENTENCE_TIMEOUT = 0.75  # 0.75 seconds to consider a word continuous

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

def predict_sign(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

def start_camera_thread():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        threading.Thread(target=gen_frames).start()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        start_camera_thread()
        return jsonify({'message': 'Camera started'})
    return jsonify({'message': 'Camera already running'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, running
    if cap and cap.isOpened():
        cap.release()
        cap = None
        running = False
        return jsonify({'message': 'Camera stopped'})
    return jsonify({'message': 'Camera is not running'})

def gen_frames():
    global cap, current_word, start_time, sentence, last_sentence, running
    frame_count = 0  # Add frame counter

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1  # Increment frame count
        
        # Only process every 5th frame
        if frame_count % 5 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if results.left_hand_landmarks or results.right_hand_landmarks:
                predicted_sign = predict_sign(frame)
                current_time = time.time()

                if predicted_sign == current_word:
                    if start_time and (current_time - start_time) >= SENTENCE_TIMEOUT:
                        if len(sentence) == 0 or sentence.split()[-1] != predicted_sign:
                            sentence += f"{predicted_sign} "
                        current_word = None
                else:
                    current_word = predicted_sign
                    start_time = current_time

        # Prepare the frame to display it on the website
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.03)  # Add a slight delay between frame captures to manage processing speed

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_prediction')
def current_prediction():
    global last_sentence, sentence
    return jsonify({'prediction': current_word, 'sentence': sentence, 'last_sentence': last_sentence})

#   ++++++++++++++++++++++++++++ Upload image or video +++++++++++++++++++++++++++++++++++++

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

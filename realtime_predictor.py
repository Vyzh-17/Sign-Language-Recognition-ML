import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
import pyttsx3
from collections import deque

# Load model and label encoder
model = tf.keras.models.load_model('onehand_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Sentence & state tracking
sentence = ""
last_pred = ""
gesture_hold_start = None
pred_buffer = deque(maxlen=5)  # Buffer to smooth predictions

# Text-to-speech
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    smoothed_pred = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            min_x, min_y = min(x_list), min(y_list)
            max_x, max_y = max(x_list), max(y_list)

            norm_x = [(x - min_x) / (max_x - min_x + 1e-6) for x in x_list]
            norm_y = [(y - min_y) / (max_y - min_y + 1e-6) for y in y_list]

            input_data = np.array([norm_x + norm_y])
            prediction = model.predict(input_data, verbose=0)
            predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

            pred_buffer.append(predicted_label)
            smoothed_pred = max(set(pred_buffer), key=pred_buffer.count)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if smoothed_pred == last_pred:
            if gesture_hold_start and (time.time() - gesture_hold_start > 1.0):
                sentence += smoothed_pred
                print(f"✔️ Added: {smoothed_pred}")
                last_pred = smoothed_pred
                gesture_hold_start = None
                pred_buffer.clear()
        else:
            last_pred = smoothed_pred
            gesture_hold_start = time.time()
    else:
        # Hand not detected – reset state so same gesture can be re-added
        last_pred = ""
        gesture_hold_start = None
        pred_buffer.clear()

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space
        sentence += ' '
    elif key == ord('c'):  # Clear all
        sentence = ""
        last_pred = ""
    elif key == 8:  # Backspace key
        sentence = sentence[:-1]
    elif key == 13:  # Enter = Speak
        if sentence:
            print("🗣️ Speaking:", sentence)
            speak(sentence)
            sentence = ""
            last_pred = ""
    elif key == ord('q'):  # Quit
        break

    # Display
    cv2.putText(frame, f'Prediction: {smoothed_pred}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.putText(frame, f'Sentence: {sentence}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Sign Language Sentence Builder", frame)

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from pygame import mixer
import os
import time  # Import the time module

# Initialize mixer and load alarm sound (use .wav file)
mixer.init()
alarm_path = r"path\to\your\.wav file"
if not os.path.exists(alarm_path):
    raise FileNotFoundError(f"Alarm sound file not found at: {alarm_path}")
mixer.music.load(alarm_path)

# Load pre-trained model
model = load_model(r"Directory\of\your\model")

# Load HaarCascade files
#Used as a reference to run the detection
face_cascade_path = "haarcascade\face\path\and\file"
eye_cascade_path = "haarcascade\eye\path\file"
if not os.path.exists(face_cascade_path) or not os.path.exists(eye_cascade_path):
    raise FileNotFoundError("HaarCascade files not found.")
face = cv2.CascadeClassifier(face_cascade_path)
eye = cv2.CascadeClassifier(eye_cascade_path)

# Initialize variables
cap = cv2.VideoCapture(0)
score = 0
alarm_playing = False
drowsy_start_time = None  # To track when drowsiness starts

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    eyes = eye.detectMultiScale(gray)

    for (x, y, w, h) in eyes:
        eye_img = gray[y:y+h, x:x+w]
        eye_img = cv2.resize(eye_img, (64, 64))
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        eye_img = eye_img / 255.0
        eye_img = eye_img.reshape(1, 64, 64, 3)  # Ensure the correct input shape

        pred = model.predict(eye_img)[0][0]

        if pred < 0.5:
            score += 1
            status = "Closed"
        else:
            score -= 1
            status = "Open"

        cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if score < 0: 
            score = 0

        if score > 10:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()  # Record the start time of drowsiness

            elapsed_time = time.time() - drowsy_start_time
            if elapsed_time > 5:  # Check if drowsiness persists for more than 5 seconds
                cv2.putText(frame, "DROWSY!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                if not alarm_playing:
                    try:
                        mixer.music.play(-1)  # Play the buzzer in a loop
                        alarm_playing = True
                    except:
                        pass
        else:
            drowsy_start_time = None  # Reset the timer if score goes below the threshold
            mixer.music.stop()
            alarm_playing = False

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



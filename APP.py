import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # Import the text-to-speech library
import pygame

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Set up the camera capture
cap = cv2.VideoCapture(0)

# Initialize Pygame
pygame.init()
welcome = pygame.mixer.Sound('initial.mp3')
thankyou = pygame.mixer.Sound('final.mp3')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Define labels dictionary
labels_dict = {0: 'HELLO', 1: 'HOW ARE YOU', 2: 'I LOVE YOU', 3: 'THANK YOU', 4: 'TEAM V S P'}
last_recognized = None  # Variable to track the last recognized gesture

# welcome message
welcome.play()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract features for model prediction
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            cx, cy = np.mean(x_), np.mean(y_)
            x_scaled = [(x - cx) * 100 for x in x_]
            y_scaled = [(y - cy) * 100 for y in y_]
            data_aux = x_scaled + y_scaled

            # Prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Drawing the bounding box and displaying the prediction
            x1, y1 = int(min(x_) * frame.shape[1]), int(min(y_) * frame.shape[0])
            x2, y2 = int(max(x_) * frame.shape[1]), int(max(y_) * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Voice output
            if predicted_character != last_recognized:
                engine.say(predicted_character)
                engine.runAndWait()
                last_recognized = predicted_character  # Update the last recognized gesture

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(25) & 0xFF == ord('f'):
        thankyou.play()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

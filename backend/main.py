from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import uvicorn
import os

app = FastAPI()

# Load Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load Model
MODEL_PATH = "model/keypoint_classifier.hdf5"
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.post("/recognize/")
async def recognize_gesture(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            landmark_flattened = np.array(landmark_list).flatten().reshape(1, -1)

            if model:
                prediction = model.predict(landmark_flattened)
                return {"gesture": str(np.argmax(prediction))}

    return {"gesture": "unknown"}

@app.get("/tts/{text}")
async def text_to_speech(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

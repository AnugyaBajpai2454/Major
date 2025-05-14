import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# ✅ Streamlit page setup
st.set_page_config(
    page_title="Smart Gesture",
    page_icon="sign_logo.jpg",  # Make sure this image file exists in the same folder
    layout="centered"
)

# ✅ Hide default Streamlit UI
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ✅ Session state for history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ✅ Title and Webcam toggle
st.title("Hand Gesture Classifier")
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

# ✅ Sidebar: Gesture history section
with st.sidebar:
    st.markdown("### 📜 Gesture History")
    for i, gesture in enumerate(reversed(st.session_state['history'])):
        st.write(f"{i+1}. {gesture}")
    if st.button("Clear History", key="clear_history"):
        st.session_state['history'] = []

# ✅ Model and camera setup
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["Hello", "I love you", "No", "Please", "Sorry", "Thank you", "yes"]
offset = 20
imgSize = 300
cap = cv2.VideoCapture(0)

# ✅ Main loop
while run:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap: wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        label = labels[index]

        # ✅ Add to history only if different from last
        if len(st.session_state['history']) == 0 or label != st.session_state['history'][-1]:
            st.session_state['history'].append(label)
            st.session_state['history'] = st.session_state['history'][-5:]

        # ✅ Display prediction on frame
        cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # ✅ Show webcam frame
    FRAME_WINDOW.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))

cap.release()


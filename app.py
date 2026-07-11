import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp

# Assign the solutions to variables using standard dot notation
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# High confidence thresholds to block cross-talk and jittering
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,             
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75   
)

face_detector = mp_face.FaceDetection(
    min_detection_confidence=0.80  
)

# Using VideoProcessorBase instead of the deprecated VideoTransformerBase
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Grab raw un-flipped frame from the webcam
        img = frame.to_ndarray(format="bgr24")
        h_img, w_img, _ = img.shape

        # Convert to RGB for native MediaPipe calculation
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run the models on the raw data
        face_results = face_detector.process(rgb_img)
        hand_results = hands.process(rgb_img)

        # Mirror the camera image
        img = cv2.flip(img, 1)

        # Draw Face Detections
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                raw_x = int(bbox.xmin * w_img)
                y = int(bbox.ymin * h_img)
                w = int(bbox.width * w_img)
                h = int(bbox.height * h_img)

                x = w_img - raw_x - w

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Hand Skeletons
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmark.x = 1.0 - landmark.x
                
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Return a standard video frame
        return frame.from_ndarray(img, format="bgr24")

# --- STREAMLIT UI ---
st.title("TinkCode Face Detection & Hand Tracking WebApp")
st.write("Click 'Start' below to allow camera access and begin tracking in real time.")

webrtc_streamer(
    key="vision-tracking",
    video_transformer_factory=VideoProcessor,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            # Standard Google public STUN fallbacks
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            
            # Open Relay Project Free TURN Server Configuration
            {
                "urls": ["turn:staticauth.openrelay.metered.ca:80", "turn:staticauth.openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayprojectsecret"
            },
            {
                "urls": ["turns:staticauth.openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayprojectsecret"
            }
        ]
    }
)
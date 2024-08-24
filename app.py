from flask import Flask, Response, jsonify
import cv2
import dlib
import face_recognition
import numpy as np
import time
from inference_sdk import InferenceHTTPClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# Load models and data
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

passport_image_path = 'passport_size.jpg'
passport_image = face_recognition.load_image_file(passport_image_path)
passport_image_encoding = face_recognition.face_encodings(passport_image)[0]

# Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="HGL0g1vvqgjMRMpQqPjF"
)

status = {
    "person_verification": "Unknown",
    "talking": "Not Talking",
    "multiple_people": "Single Person",
    "phone_usage": "No Phone Detected"
}

def get_mouth_aspect_ratio(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[9])
    B = np.linalg.norm(mouth_points[4] - mouth_points[7])
    C = np.linalg.norm(mouth_points[0] - mouth_points[6])
    mar = (A + B) / (2.0 * C)
    return mar

def resize_frame(frame, target_size=(320, 240)):
    return cv2.resize(frame, target_size)

def process_frame(frame):
    global status
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Face recognition and talking detection
    for face in faces:
        face_encodings = face_recognition.face_encodings(frame, [(face.top(), face.right(), face.bottom(), face.left())])
        if face_encodings:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces([passport_image_encoding], face_encoding)
            if matches[0]:
                status["person_verification"] = "Verified"
            else:
                status["person_verification"] = "Not Verified"

        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        mouth_points = landmarks[48:68]
        mar = get_mouth_aspect_ratio(mouth_points)
        if mar > 0.5:
            status["talking"] = "Talking"
        else:
            status["talking"] = "Not Talking"

    # Multiple people detection
    multi_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(multi_faces) >= 2:
        status["multiple_people"] = "Multiple People Detected"
    else:
        status["multiple_people"] = "Single Person"

    # Mobile phone detection
    resized_frame = resize_frame(frame, target_size=(320, 240))
    result = CLIENT.infer(resized_frame, model_id="mobile_phone_detection/4")
    if len(result['predictions']) == 0:
        status["phone_usage"] = "No Phone Detected"
    else:
        status["phone_usage"] = "Phone Detected"
        for pred in result['predictions']:
            x_min = int(pred['x'] * frame.shape[1] / resized_frame.shape[1])
            y_min = int(pred['y'] * frame.shape[0] / resized_frame.shape[0])
            x_max = int((pred['x'] + pred['width']) * frame.shape[1] / resized_frame.shape[1])
            y_max = int((pred['y'] + pred['height']) * frame.shape[0] / resized_frame.shape[0])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduce height
    frame_interval = 2  # Process every nth frame
    last_frame_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            last_frame_time = current_time
            frame = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/monitor')
def get_status():
    return jsonify(status)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

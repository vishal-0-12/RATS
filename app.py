from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, session
import os
import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import csv
from sklearn.neighbors import KDTree
from datetime import datetime
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Load YOLOv5 model
model = YOLO('yolov5/yolov5s.pt')  # Use 'yolov5s' or your custom-trained YOLOv5 model

# Load user credentials from a CSV file
users = {}
with open('users.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        username = row['username']
        user_id = row['user_id']
        users[username] = user_id

# Load known faces and their names and IDs from CSV file
known_face_encodings = []
known_face_names = []
known_face_ids = []

with open('known_faces.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image_path = row['image_path']
        name = row['name']
        id = row['id']

        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)
            known_face_ids.append(id)

# Build a KD-tree for fast nearest neighbor search
face_encodings_tree = KDTree(known_face_encodings)

# Global variables to store attendance data
attendance_data = []
recorded_names = set()
attendance_filename = ""

@app.route('/')
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_user():
    username = request.form['username']
    user_id = request.form['user_id']
    if username in users and users[username] == user_id:
        session['username'] = username
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error="Invalid username or ID. Please try again.")

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])

@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    if 'username' not in session:
        return redirect(url_for('login'))
    global attendance_filename, recorded_names, attendance_data
    if request.method == 'POST':
        teacher_name = request.form['teacher_name']
        class_name = request.form['class_name']
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        attendance_folder = f"attendance/{teacher_name}/{class_name}/{current_time[:8]}"
        os.makedirs(attendance_folder, exist_ok=True)
        attendance_filename = f"{attendance_folder}/{teacher_name}_{class_name}_{current_time}.csv"

        # Reset attendance data
        recorded_names = set()
        attendance_data = []

        return render_template('take_attendance.html', teacher_name=teacher_name, class_name=class_name, attendance_data=attendance_data)
    return render_template('take_attendance.html')

@app.route('/get_attendance')
def get_attendance():
    if 'username' not in session:
        return redirect(url_for('login'))
    global attendance_data
    return jsonify(attendance_data)

def generate_frames():
    global attendance_data, recorded_names, attendance_filename
    cap = cv2.VideoCapture(0)  # Change to 0 if 1 does not work

    # Set the frame width, height, and frame rate
    frame_width = 1280
    frame_height = 720
    frame_rate = 30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects with YOLOv5
        results = model(frame)

        # Apply non-maximum suppression to remove overlapping boxes
        boxes = []
        confidences = []
        class_ids = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = box.cls[0]

                if confidence > 0.7 and class_id == 0:  # class_id == 0 for face
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

        for i in indices:
            box = boxes[i]
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h

            # Extract the face ROI
            face_roi = frame[y1:y2, x1:x2]

            # Convert the face ROI to RGB (face_recognition uses RGB format)
            rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Recognize the face
            face_encodings = face_recognition.face_encodings(rgb_face_roi)
            if not face_encodings:
                continue  # Skip if no face encodings are found

            face_names = []
            face_ids = []
            for face_encoding in face_encodings:
                # Find the closest match in the KD-tree
                dist, ind = face_encodings_tree.query([face_encoding], k=1)
                best_match_index = ind[0][0]
                if dist[0][0] < 0.7:  # Adjust the threshold as needed
                    name = known_face_names[best_match_index]
                    id = known_face_ids[best_match_index]
                else:
                    name = "Unknown"
                    id = "Unknown"

                face_names.append(name)
                face_ids.append(id)

            # Detect hand gestures
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)

            hand_detected = False
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_detected = True

            # Draw bounding box and label
            if face_names and hand_detected:
                label = f"{face_names[0]} ({face_ids[0]})"
                # Write attendance data to CSV file if not already recorded and not "Unknown"
                if face_names[0] != "Unknown" and face_names[0] not in recorded_names:
                    attendance_data.append({'Name': face_names[0], 'ID': face_ids[0], 'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    recorded_names.add(face_names[0])
            else:
                label = "Unknown"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    hands.close()

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_attendance')
def save_attendance():
    if 'username' not in session:
        return redirect(url_for('login'))
    global attendance_filename, attendance_data
    if attendance_filename and attendance_data:
        with open(attendance_filename, 'w', newline='') as csvfile:
            fieldnames = ['Name', 'ID', 'Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in attendance_data:
                writer.writerow(data)
    return redirect(url_for('home'))

@app.route('/see_attendance', methods=['GET', 'POST'])
def see_attendance():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        teacher_name = request.form['teacher_name']
        class_name = request.form['class_name']
        date = request.form['date']
        attendance_folder = f"attendance/{teacher_name}/{class_name}/{date}"
        if os.path.exists(attendance_folder):
            csv_files = os.listdir(attendance_folder)
            return render_template('see_attendance.html', teacher_name=teacher_name, class_name=class_name, date=date, csv_files=csv_files)
    return render_template('see_attendance.html', csv_files=[])

@app.route('/view_attendance', methods=['POST'])
def view_attendance():
    if 'username' not in session:
        return redirect(url_for('login'))
    teacher_name = request.form['teacher_name']
    class_name = request.form['class_name']
    date = request.form['date']
    csv_file = request.form.get('csv_file')
    attendance_records = []
    if csv_file:
        file_path = f"attendance/{teacher_name}/{class_name}/{date}/{csv_file}"
        if os.path.exists(file_path):
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    attendance_records.append(row)
    return render_template('see_attendance.html', teacher_name=teacher_name, class_name=class_name, date=date, csv_file=csv_file, attendance_records=attendance_records, csv_files=os.listdir(f"attendance/{teacher_name}/{class_name}/{date}"))

if __name__ == '__main__':
    app.run(debug=True)
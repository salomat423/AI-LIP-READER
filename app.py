from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
import bcrypt
import csv
import os
import threading

import cv2
import dlib
import face_recognition
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = "s"

# Absolute path to this folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize variables for recording
recording = False
recording2 = False

# Live status for UI polling
last_face_detected = False
last_frame_lock = threading.Lock()

# Load face detector
detector = dlib.get_frontal_face_detector()

# Load facial landmarks predictor
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(
        f"Missing dlib predictor file: {PREDICTOR_PATH}\n"
        "Download 'shape_predictor_68_face_landmarks.dat' and place it next to app.py."
    )
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Initialize lip offsets and base positions
lip_offsets = []
lip_offsets2 = []
base_positions = None
base_positions2 = None
previous_positions = None
previous_positions2 = None

# Directory to save captured images
capture_dir = "captured_faces"
os.makedirs(capture_dir, exist_ok=True)

# Absolute path to users.csv
USERS_CSV_PATH = os.path.join(BASE_DIR, "users.csv")


def cosine_similarity(seq1, seq2):
    seq1_flat = (seq1 - seq1[0]).flatten()
    seq2_flat = (seq2 - seq2[0]).flatten()

    if np.linalg.norm(seq1_flat) == 0 or np.linalg.norm(seq2_flat) == 0:
        return 0.0

    return float(np.dot(seq1_flat, seq2_flat) / (np.linalg.norm(seq1_flat) * np.linalg.norm(seq2_flat)))


def lips_are_moving(current_positions, previous_positions, threshold=3.5):
    if previous_positions is None:
        return False
    distances = [
        np.linalg.norm(np.array(current_positions[i]) - np.array(previous_positions[i]))
        for i in range(len(current_positions))
    ]
    return any(distance > threshold for distance in distances)


def gen():
    global recording, recording2
    global lip_offsets, lip_offsets2
    global base_positions, base_positions2
    global previous_positions, previous_positions2
    global last_face_detected

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally to make it non-mirrored
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        with last_frame_lock:
            last_face_detected = bool(faces)

        if faces:
            landmarks = predictor(gray, faces[0])
            lips = landmarks.parts()[48:68]
            current_positions = [(p.x, p.y) for p in lips]

            if recording:
                if base_positions is None:
                    base_positions = current_positions
                if lips_are_moving(current_positions, previous_positions):
                    current_offsets = [
                        (p.x - base_positions[i][0], p.y - base_positions[i][1]) for i, p in enumerate(lips)
                    ]
                    lip_offsets.append(current_offsets)
                previous_positions = current_positions

            if recording2:
                if base_positions2 is None:
                    base_positions2 = current_positions
                if lips_are_moving(current_positions, previous_positions2):
                    current_offsets2 = [
                        (p.x - base_positions2[i][0], p.y - base_positions2[i][1]) for i, p in enumerate(lips)
                    ]
                    lip_offsets2.append(current_offsets2)
                previous_positions2 = current_positions

            # Draw the alignment guide (rectangle)
            x_min = min(p.x for p in lips) - 10
            x_max = max(p.x for p in lips) + 10
            y_min = min(p.y for p in lips) - 10
            y_max = max(p.y for p in lips) + 10
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            for point in lips:
                cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

        status = "Recording1" if recording else "Not Recording1"
        status2 = "Recording2" if recording2 else "Not Recording2"
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if recording else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status2,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if recording2 else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


def save_to_csv(filename, offsets):
    if offsets:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Point", "X", "Y"])
            for frame, off in enumerate(offsets):
                for point, (x, y) in enumerate(off):
                    writer.writerow([frame, point, x, y])


def compare_faces(image_path1, image_path2):
    try:
        image1 = face_recognition.load_image_file(image_path1)
        image2 = face_recognition.load_image_file(image_path2)

        encodings1 = face_recognition.face_encodings(image1)
        encodings2 = face_recognition.face_encodings(image2)

        if not encodings1 or not encodings2:
            return False

        match_results = face_recognition.compare_faces([encodings1[0]], encodings2[0])
        return bool(match_results[0])
    except Exception:
        return False


def save_user_to_csv(username, hashed_password):
    with open(USERS_CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([username, hashed_password])


def user_exists(username):
    try:
        with open(USERS_CSV_PATH, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == username:
                    return True
    except FileNotFoundError:
        pass
    return False


def verify_password(username, password):
    try:
        with open(USERS_CSV_PATH, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == username:
                    return bcrypt.checkpw(password.encode("utf-8"), row[1].encode("utf-8"))
    except FileNotFoundError:
        pass
    return False


def wants_json_response():
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept


@app.route("/")
def landing():
    return render_template("landing.html", username=session.get("username"))


@app.route("/registration")
def registration():
    return render_template("registration.html", username=session.get("username"))


@app.route("/account")
def account():
    msg = request.args.get("msg") or ""
    return render_template("account.html", username=session.get("username"), msg=msg)


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not username:
        return "Username is required!", 400
    if password != confirm_password:
        return "Passwords do not match!", 400
    if user_exists(username):
        return "User already exists!", 400

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    save_user_to_csv(username, hashed_password.decode("utf-8"))
    session["username"] = username
    return redirect(url_for("registration"))


@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/login", methods=["POST"])
def login_post():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if user_exists(username) and verify_password(username, password):
        session["username"] = username
        return redirect(url_for("record_login"))
    return "Invalid username or password!", 401


@app.route("/record_login")
def record_login():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("record_login.html", username=session.get("username"))


@app.route("/stop_recording_login", methods=["POST"])
def stop_recording_login():
    global recording2, lip_offsets2, base_positions2, previous_positions2
    if "username" not in session:
        return redirect(url_for("login"))

    recording2 = False
    username = session["username"]

    user_capture_dir = os.path.join(capture_dir, username)
    os.makedirs(user_capture_dir, exist_ok=True)

    face_image_path = os.path.join(user_capture_dir, "face_login.jpg")
    lip_offsets_path = os.path.join(user_capture_dir, "lip_offsets_login.csv")

    save_to_csv(lip_offsets_path, lip_offsets2)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(face_image_path, frame)
    cap.release()

    face_registration_path = os.path.join(user_capture_dir, "face.jpg")
    if not compare_faces(face_registration_path, face_image_path):
        if wants_json_response():
            return jsonify({"ok": False, "error": "Face verification failed!"}), 401
        return "Face verification failed!", 401

    registration_offsets_path = os.path.join(user_capture_dir, "lip_offsets.csv")
    if not os.path.exists(registration_offsets_path):
        if wants_json_response():
            return jsonify({"ok": False, "error": "Registration lip data not found!"}), 400
        return "Registration lip data not found!", 400

    registered_offsets = pd.read_csv(registration_offsets_path)
    registration_offsets = registered_offsets.drop(columns=["Frame", "Point"]).to_numpy().reshape((-1, 20, 2))
    similarity = cosine_similarity(np.array(lip_offsets2), registration_offsets)
    if similarity <= 0.8:
        if wants_json_response():
            return jsonify({"ok": False, "error": "Lip movements do not match!", "similarity": similarity}), 401
        return "Lip movements do not match!", 401

    redirect_url = url_for("account", msg="Access granted")
    if wants_json_response():
        return jsonify({"ok": True, "redirect": redirect_url})
    return redirect(redirect_url)


@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    global recording, lip_offsets, base_positions, previous_positions
    if "username" not in session:
        return redirect(url_for("login"))

    recording = False
    username = session["username"]

    user_capture_dir = os.path.join(capture_dir, username)
    os.makedirs(user_capture_dir, exist_ok=True)

    face_image_path = os.path.join(user_capture_dir, "face.jpg")
    lip_offsets_path = os.path.join(user_capture_dir, "lip_offsets.csv")

    save_to_csv(lip_offsets_path, lip_offsets)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(face_image_path, frame)
    cap.release()

    if wants_json_response():
        return jsonify({"ok": True})
    return "Recording stopped and data saved!"


@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording, lip_offsets, base_positions, previous_positions
    if "username" not in session:
        return redirect(url_for("login"))
    recording = True
    lip_offsets = []
    base_positions = None
    previous_positions = None
    if wants_json_response():
        return jsonify({"ok": True})
    return "Recording started!"


@app.route("/start_recording_login", methods=["POST"])
def start_recording_login():
    global recording2, lip_offsets2, base_positions2, previous_positions2
    if "username" not in session:
        return redirect(url_for("login"))
    recording2 = True
    lip_offsets2 = []
    base_positions2 = None
    previous_positions2 = None
    if wants_json_response():
        return jsonify({"ok": True})
    return "Recording started!"


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with last_frame_lock:
        face = bool(last_face_detected)
    return jsonify({"faceDetected": face, "recording": bool(recording), "recording2": bool(recording2)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import os
import pickle
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

ATTENDANCE_FILE = 'Attendance.csv'
ADMIN_FILE = 'admin_users.pkl'

# Ensure attendance file exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Time', 'Date'])

# Ensure admin file exists
if not os.path.exists(ADMIN_FILE):
    with open(ADMIN_FILE, 'wb') as f:
        pickle.dump({'admin': generate_password_hash('admin123')}, f)

@app.route('/', methods=['GET', 'POST'])
def public():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if os.path.exists(ADMIN_FILE):
            with open(ADMIN_FILE, 'rb') as f:
                users = pickle.load(f)
        else:
            users = {}

        if username in users:
            if check_password_hash(users[username], password):
                session['username'] = username
                return redirect(url_for('index'))
            else:
                flash('❌ Incorrect password.', 'error')
        else:
            flash('❌ Username not found.', 'error')

        return redirect(url_for('public'))

    return render_template('public.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('public'))

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('public'))

    today = datetime.now().strftime('%d-%m-%Y')
    attendance = []
    count = 0
    total = 0

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            for row in all_rows:
                if row['Date'] == today:
                    attendance.append(row)
                    count += 1
        names_today = set([row['Name'] for row in attendance])
        total = len(set([row['Name'] for row in all_rows]))

    return render_template('index.html', session=session, attendance=attendance, count=count, total=total)

@app.route('/download_csv')
def download_csv():
    if 'username' not in session:
        return redirect(url_for('public'))
    return send_file(ATTENDANCE_FILE, as_attachment=True)

@app.route('/mark_manual_attendance', methods=['POST'])
def mark_manual_attendance():
    if 'username' not in session:
        return redirect(url_for('public'))

    name = request.form['manual_name'].strip()
    now = datetime.now()
    time = now.strftime('%H:%M:%S')
    date = now.strftime('%d-%m-%Y')

    already_marked = False
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Name'] == name and row['Date'] == date:
                    already_marked = True
                    break

    if not already_marked:
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, time, date])

    return redirect(url_for('index'))

@app.route('/mark_attendance_from_browser', methods=['POST'])
def mark_attendance_from_browser():
    import base64
    import face_recognition
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO

    image_data = request.json['image']
    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (ValueError, UnidentifiedImageError) as e:
        print("❌ Image decoding failed:", e)
        return {'status': 'fail', 'error': 'Invalid image format'}

    image_np = np.array(image)

    path = 'images'
    images = []
    classNames = []

    for filename in os.listdir(path):
        img = face_recognition.load_image_file(f'{path}/{filename}')
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])

    # ✅ FIXED: Safe face encoding loading
    encodings = []
    for img in images:
        face_enc = face_recognition.face_encodings(img)
        if face_enc:
            encodings.append(face_enc[0])

    faces_in_frame = face_recognition.face_locations(image_np)
    encodings_in_frame = face_recognition.face_encodings(image_np, faces_in_frame)

    for encode_face, face_loc in zip(encodings_in_frame, faces_in_frame):
        matches = face_recognition.compare_faces(encodings, encode_face)
        face_distances = face_recognition.face_distance(encodings, encode_face)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = classNames[best_match_index]

            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            date = now.strftime('%d-%m-%Y')

            already_marked = False
            with open(ATTENDANCE_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Name'] == name and row['Date'] == date:
                        already_marked = True
                        break

            if not already_marked:
                with open(ATTENDANCE_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time, date])

            return {'status': 'success', 'name': name}

    return {'status': 'fail'}

@app.route('/register_face', methods=['POST'])
def register_face():
    import base64
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO

    name = request.form['name'].strip()
    image_data = request.form['image']
    if not name or not image_data:
        return redirect(url_for('index'))

    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (ValueError, UnidentifiedImageError) as e:
        print("❌ Failed to register face:", e)
        flash("Image upload failed. Try again.", "error")
        return redirect(url_for('index'))

    image_np = np.array(image)

    path = 'images'
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.imwrite(f'{path}/{name}.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

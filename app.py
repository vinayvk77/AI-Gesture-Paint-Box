from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import time
import json
from collections import deque

app = Flask(__name__)
socketio = SocketIO(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/artworks'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global variables
canvas = None
x1, y1 = 0, 0
color = (0, 0, 255)  # Default color (red)
color_name = "Red"
brush_size = 5
eraser_size = 20
mode = "draw"
colors = {
    "#FF0000": "Red",
    "#00FF00": "Green",
    "#0000FF": "Blue",
    "#FFFF00": "Yellow",
    "#FF00FF": "Magenta",
    "#00FFFF": "Cyan",
    "#FFFFFF": "White",
    "#000000": "Black",
    "#FFA500": "Orange",
    "#800080": "Purple",
    "#008000": "Dark Green",
    "#FFC0CB": "Pink"
}
color_index = 0
text_mode = False
text_content = ""
points = deque(maxlen=5)
saved_artworks = []
last_save_time = 0

class Artwork:
    def __init__(self, filename, title, artist, date):
        self.filename = filename
        self.title = title
        self.artist = artist
        self.date = date

def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_status = {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}
    
    if landmarks[4].x < landmarks[3].x:
        finger_status['thumb'] = 1
    
    for i, tip in enumerate(finger_tips[1:]):
        if landmarks[tip].y < landmarks[tip-2].y:
            finger_status[list(finger_status.keys())[i+1]] = 1
    
    return finger_status

def select_mode(finger_status):
    global mode, color_index, text_mode, text_content
    
    if finger_status['thumb'] == 1 and sum(finger_status.values()) == 1:
        return "clear"
    
    if finger_status['index'] == 1 and sum(finger_status.values()) == 1:
        text_mode = False
        return "draw"
    
    if finger_status['index'] == 1 and finger_status['middle'] == 1 and sum(finger_status.values()) == 2:
        text_mode = False
        return "select"
    
    if (finger_status['index'] == 1 and finger_status['middle'] == 1 and 
        finger_status['ring'] == 1 and sum(finger_status.values()) == 3):
        text_mode = True
        text_content = ""
        socketio.emit('message', {'type': 'text_mode'})
        return "text"
    
    if sum(finger_status.values()) == 0:
        text_mode = False
        return "erase"
    
    return mode

def generate_frames():
    global canvas, x1, y1, color, brush_size, eraser_size, mode, color_index, text_mode, text_content, points, color_name
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        if canvas is None:
            canvas = np.zeros_like(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks (optional, for visualization)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                
                finger_status = count_fingers(hand_landmarks.landmark)
                new_mode = select_mode(finger_status)
                
                if new_mode == "clear":
                    canvas = np.zeros_like(canvas)
                    mode = "draw"
                    socketio.emit('message', {'type': 'mode_update', 'mode': 'Draw'})
                elif new_mode == "select":
                    color_index = (color_index + 1) % len(colors)
                    hex_color = list(colors.keys())[color_index]
                    color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))[::-1]  # Convert hex to BGR
                    color_name = colors[hex_color]
                    mode = "draw"
                    socketio.emit('message', {
                        'type': 'color_update',
                        'color_name': color_name,
                        'color': hex_color
                    })
                else:
                    if mode != new_mode:
                        mode = new_mode
                        socketio.emit('message', {'type': 'mode_update', 'mode': mode.capitalize()})
                
                index_tip = hand_landmarks.landmark[8]
                x2, y2 = int(index_tip.x * width), int(index_tip.y * height)
                
                if mode == "draw":
                    points.append((x2, y2))
                    if len(points) >= 2:
                        for i in range(1, len(points)):
                            cv2.line(canvas, points[i-1], points[i], color, brush_size)
                    x1, y1 = x2, y2
                
                elif mode == "erase":
                    cv2.circle(canvas, (x2, y2), eraser_size, (0, 0, 0), -1)
                
                elif mode == "text" and text_mode:
                    # Text mode is handled through the web interface
                    pass
        
        # Draw UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        mode_text = f"Mode: {mode.capitalize()}"
        color_text = f"Color: {color_name}"
        size_text = f"Brush: {'Small' if brush_size == 3 else 'Medium' if brush_size == 5 else 'Large'}"
        
        cv2.putText(frame, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, color_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Merge canvas with frame
        frame = cv2.add(frame, canvas)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Load saved artworks from directory
    global saved_artworks
    saved_artworks = []
    
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith('.png'):
                # Extract metadata from filename
                parts = filename.split('_')
                if len(parts) >= 4:
                    timestamp = parts[2] + '_' + parts[3].split('.')[0]
                    try:
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        date_str = dt.strftime("%B %d, %Y at %H:%M")
                        title = ' '.join(parts[1:-2]) or "Untitled"
                        artist = "Anonymous"
                        
                        # Check for metadata file
                        meta_file = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.meta')
                        if os.path.exists(meta_file):
                            with open(meta_file, 'r') as f:
                                meta = json.load(f)
                                title = meta.get('title', title)
                                artist = meta.get('artist', artist)
                        
                        saved_artworks.append(Artwork(filename, title, artist, date_str))
                    except ValueError:
                        continue
    
    # Sort by date (newest first)
    saved_artworks.sort(key=lambda x: x.filename, reverse=True)
    return render_template('index.html', saved_artworks=saved_artworks)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save', methods=['POST'])
def save_artwork():
    global canvas, saved_artworks, last_save_time
    
    current_time = time.time()
    if current_time - last_save_time < 300:  # 5 minutes
        return jsonify({'status': 'error', 'message': 'Please wait 5 minutes between saves'})
    
    data = request.get_json()
    title = data.get('title', 'Untitled')
    artist = data.get('artist', 'Anonymous')
    
    if canvas is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"artwork_{title.replace(' ', '_')}_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, canvas)
        
        # Save metadata
        meta = {
            'title': title,
            'artist': artist
        }
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename + '.meta'), 'w') as f:
            json.dump(meta, f)
        
        date = datetime.now().strftime("%B %d, %Y at %H:%M")
        artwork = Artwork(filename, title, artist, date)
        saved_artworks.append(artwork)
        
        last_save_time = current_time
        return jsonify({'status': 'success', 'filename': filename})
    
    return jsonify({'status': 'error', 'message': 'No canvas to save'})

@app.route('/get_artworks')
def get_artworks():
    artworks_data = []
    for artwork in saved_artworks:
        artworks_data.append({
            'filename': artwork.filename,
            'title': artwork.title,
            'artist': artwork.artist,
            'date': artwork.date
        })
    return jsonify(artworks_data)

@app.route('/get_artwork_info')
def get_artwork_info():
    filename = request.args.get('filename')
    for artwork in saved_artworks:
        if artwork.filename == filename:
            return jsonify({
                'filename': artwork.filename,
                'title': artwork.title,
                'artist': artwork.artist,
                'date': artwork.date
            })
    return jsonify({'status': 'error', 'message': 'Artwork not found'})

@app.route('/delete_artwork', methods=['POST'])
def delete_artwork():
    data = request.get_json()
    filename = data.get('filename')
    
    for i, artwork in enumerate(saved_artworks):
        if artwork.filename == filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            # Remove metadata file
            meta_file = os.path.join(app.config['UPLOAD_FOLDER'], filename + '.meta')
            if os.path.exists(meta_file):
                os.remove(meta_file)
            saved_artworks.pop(i)
            return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Artwork not found'})

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    global canvas
    canvas = None
    return jsonify({'status': 'success'})

@app.route('/set_color', methods=['POST'])
def set_color():
    global color, color_name, color_index
    data = request.get_json()
    hex_color = data.get('color')
    
    if hex_color in colors:
        color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))[::-1]  # Convert hex to BGR
        color_name = colors[hex_color]
        color_index = list(colors.keys()).index(hex_color)
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Invalid color'})

@app.route('/set_brush_size', methods=['POST'])
def set_brush_size():
    global brush_size
    data = request.get_json()
    size = data.get('size', 5)
    
    if size in [3, 5, 10]:
        brush_size = size
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Invalid brush size'})

@app.route('/submit_text', methods=['POST'])
def submit_text():
    global text_mode, text_content, color
    data = request.get_json()
    text = data.get('text', '')
    
    if text and canvas is not None:
        height, width, _ = canvas.shape
        cv2.putText(canvas, text, (width//2, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, color, 3, cv2.LINE_AA)
    
    text_mode = False
    return jsonify({'status': 'success'})

@app.route('/cancel_text', methods=['POST'])
def cancel_text():
    global text_mode
    text_mode = False
    return jsonify({'status': 'success'})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
    

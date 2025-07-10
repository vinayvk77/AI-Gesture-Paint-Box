import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
canvas = None
x1, y1 = 0, 0  # Previous finger position
color = (0, 0, 255)  # Default color (red)
brush_size = 5
eraser_size = 20
mode = "draw"  # Can be "draw", "erase", "text", "shape"
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), 
          (255, 255, 0), (255, 0, 255), (255, 255, 255)]  # RGB colors
color_index = 0
shapes = ["rectangle", "circle", "line"]
shape_index = 0
drawing = False
start_point = (0, 0)
text_mode = False
text_content = ""
font_scale = 1
font_thickness = 2
history = []
history_index = -1
max_history = 10

# Deque to store points for smoothing drawings
points = deque(maxlen=5)

def count_fingers(landmarks):
    """
    Count the number of fingers that are up
    Returns a dictionary with finger status (1 for up, 0 for down)
    """
    finger_tips = [4, 8, 12, 16, 20]  # Landmark indices for finger tips
    finger_status = {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}
    
    # Thumb (special case - compare x coordinate)
    if landmarks[4].x < landmarks[3].x:
        finger_status['thumb'] = 1
    
    # Other fingers (compare y coordinate)
    for i, tip in enumerate(finger_tips[1:]):
        if landmarks[tip].y < landmarks[tip-2].y:
            finger_status[list(finger_status.keys())[i+1]] = 1
    
    return finger_status

def select_mode(finger_status):
    """
    Determine the mode based on finger status
    """
    global mode, color_index, shape_index, text_mode, text_content
    
    # Thumb up - clear screen
    if finger_status['thumb'] == 1 and sum(finger_status.values()) == 1:
        return "clear"
    
    # Index finger up - draw mode
    if finger_status['index'] == 1 and sum(finger_status.values()) == 1:
        text_mode = False
        return "draw"
    
    # Index and middle up - color selection
    if finger_status['index'] == 1 and finger_status['middle'] == 1 and sum(finger_status.values()) == 2:
        text_mode = False
        return "select"
    
    # Index, middle, and ring up - shape selection
    if (finger_status['index'] == 1 and finger_status['middle'] == 1 and 
        finger_status['ring'] == 1 and sum(finger_status.values()) == 3):
        text_mode = False
        return "shape"
    
    # Index, middle, ring, and pinky up - text mode
    if (finger_status['index'] == 1 and finger_status['middle'] == 1 and 
        finger_status['ring'] == 1 and finger_status['pinky'] == 1 and sum(finger_status.values()) == 4):
        text_mode = True
        text_content = ""
        return "text"
    
    # Fist (no fingers up) - erase mode
    if sum(finger_status.values()) == 0:
        text_mode = False
        return "erase"
    
    # Palm open (all fingers up) - undo/redo
    if sum(finger_status.values()) >= 5:
        text_mode = False
        return "history"
    
    return mode  # Default to current mode

def draw_text(img, text, position, color=(255, 255, 255), scale=1, thickness=2):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_ui(frame, width, height):
    """Draw the user interface elements"""
    # Create a semi-transparent overlay for UI
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Display mode and color information
    mode_text = f"Mode: {mode.capitalize()}"
    color_text = f"Color: RGB{color}"
    draw_text(frame, mode_text, (20, 30), (255, 255, 255))
    draw_text(frame, color_text, (20, 60), color)
    
    if mode == "shape":
        shape_text = f"Shape: {shapes[shape_index]}"
        draw_text(frame, shape_text, (20, 90), (255, 255, 255))
    
    # Display current brush/eraser size
    size_text = f"Brush: {brush_size}px" if mode == "draw" else f"Eraser: {eraser_size}px"
    draw_text(frame, size_text, (width - 150, 30), (255, 255, 255))
    
    # Display quick instructions
    instructions = [
        "Index finger - Draw",
        "Index+Middle - Color",
        "Index+Middle+Ring - Shape",
        "All fingers - Undo/Redo",
        "Fist - Erase",
        "Thumb - Clear"
    ]
    
    for i, text in enumerate(instructions):
        draw_text(frame, text, (width // 2 - 100, 30 + i * 15), (200, 200, 200), 0.5)

def save_to_history():
    """Save current canvas state to history"""
    global history, history_index
    
    # If we're not at the end of history, truncate future history
    if history_index < len(history) - 1:
        history = history[:history_index + 1]
    
    # Save a copy of the current canvas
    history.append(canvas.copy())
    history_index += 1
    
    # Limit history size
    if len(history) > max_history:
        history.pop(0)
        history_index -= 1

def undo():
    """Undo the last action"""
    global history_index, canvas
    
    if history_index > 0:
        history_index -= 1
        canvas = history[history_index].copy()

def redo():
    """Redo the last undone action"""
    global history_index, canvas
    
    if history_index < len(history) - 1:
        history_index += 1
        canvas = history[history_index].copy()

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Initialize canvas on first frame
    if canvas is None:
        canvas = np.zeros_like(frame)
        save_to_history()
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get finger status
            finger_status = count_fingers(hand_landmarks.landmark)
            
            # Select mode based on finger status
            new_mode = select_mode(finger_status)
            
            # Handle mode changes
            if new_mode == "clear":
                save_to_history()
                canvas = np.zeros_like(canvas)
                mode = "draw"
            elif new_mode == "select":
                color_index = (color_index + 1) % len(colors)
                color = colors[color_index]
                mode = "draw"  # Return to draw mode after selection
            elif new_mode == "shape":
                shape_index = (shape_index + 1) % len(shapes)
                mode = "shape"
            elif new_mode == "history":
                # Palm open - toggle between undo and redo
                if finger_status['thumb'] == 1:
                    redo()
                else:
                    undo()
                mode = "draw"
            else:
                mode = new_mode
            
            # Get index finger tip coordinates
            index_tip = hand_landmarks.landmark[8]
            x2, y2 = int(index_tip.x * width), int(index_tip.y * height)
            
            # Draw landmarks (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if mode == "draw":
                # Store current point for smoothing
                points.append((x2, y2))
                
                # Draw smooth lines between points
                if len(points) >= 2:
                    save_to_history()
                    for i in range(1, len(points)):
                        cv2.line(canvas, points[i-1], points[i], color, brush_size)
                
                # Update previous position
                x1, y1 = x2, y2
                
            elif mode == "erase":
                save_to_history()
                cv2.circle(canvas, (x2, y2), eraser_size, (0, 0, 0), -1)
            
            elif mode == "shape":
                if finger_status['index'] == 1 and not drawing:
                    drawing = True
                    start_point = (x2, y2)
                
                if drawing:
                    if shapes[shape_index] == "rectangle":
                        cv2.rectangle(canvas, start_point, (x2, y2), color, brush_size)
                    elif shapes[shape_index] == "circle":
                        radius = int(np.sqrt((x2 - start_point[0])**2 + (y2 - start_point[1])**2))
                        cv2.circle(canvas, start_point, radius, color, brush_size)
                    elif shapes[shape_index] == "line":
                        cv2.line(canvas, start_point, (x2, y2), color, brush_size)
                
                if finger_status['index'] == 0 and drawing:
                    drawing = False
                    save_to_history()
            
            elif mode == "text" and text_mode:
                # Display text input prompt
                draw_text(frame, "Enter text (press Enter when done):", (width//4, height//2), (255, 255, 255))
                
                # Get keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    text_mode = False
                    save_to_history()
                    cv2.putText(canvas, text_content, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale, color, font_thickness, cv2.LINE_AA)
                elif key == 8:  # Backspace
                    text_content = text_content[:-1]
                elif 32 <= key <= 126:  # Printable ASCII characters
                    text_content += chr(key)
                
                # Display current text
                if text_content:
                    draw_text(frame, text_content, (x2, y2), color, font_scale, font_thickness)
    
    # Draw the UI
    draw_ui(frame, width, height)
    
    # Merge canvas with frame
    frame = cv2.add(frame, canvas)
    
    # Show the frame
    cv2.imshow('Enhanced Hand Gesture Paint', frame)
    
    # Keyboard controls for additional functionality
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        brush_size = min(brush_size + 1, 50)
        eraser_size = min(eraser_size + 5, 100)
    elif key == ord('-'):
        brush_size = max(brush_size - 1, 1)
        eraser_size = max(eraser_size - 5, 5)
    elif key == ord('s'):
        # Save the artwork
        cv2.imwrite('artwork.png', canvas)
        print("Artwork saved as 'artwork.png'")

# Release resources
cap.release()
cv2.destroyAllWindows()
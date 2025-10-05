import atexit
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os

# -------------------- CONFIG --------------------
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.1

ALPHA_FILL = 0.15
ALPHA_BORDER = 0.25
ALPHA_TEXT = 0.4

PERSIST_FRAMES = 5  # number of frames to keep highlight after missing

# -------------------- FLASK APP --------------------
app = Flask(__name__)

# -------------------- LOAD MODEL --------------------
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

# -------------------- CARD MAP --------------------
CARD_MAP = {
    "Ah": "Ace of Hearts", "Ad": "Ace of Diamonds", "As": "Ace of Spades", "Ac": "Ace of Clubs",
    "2h": "2 of Hearts", "2d": "2 of Diamonds", "2s": "2 of Spades", "2c": "2 of Clubs",
    "3h": "3 of Hearts", "3d": "3 of Diamonds", "3s": "3 of Spades", "3c": "3 of Clubs",
    "4h": "4 of Hearts", "4d": "4 of Diamonds", "4s": "4 of Spades", "4c": "4 of Clubs",
    "5h": "5 of Hearts", "5d": "5 of Diamonds", "5s": "5 of Spades", "5c": "5 of Clubs",
    "6h": "6 of Hearts", "6d": "6 of Diamonds", "6s": "6 of Spades", "6c": "6 of Clubs",
    "7h": "7 of Hearts", "7d": "7 of Diamonds", "7s": "7 of Spades", "7c": "7 of Clubs",
    "8h": "8 of Hearts", "8d": "8 of Diamonds", "8s": "8 of Spades", "8c": "8 of Clubs",
    "9h": "9 of Hearts", "9d": "9 of Diamonds", "9s": "9 of Spades", "9c": "9 of Clubs",
    "10h": "10 of Hearts", "10d": "10 of Diamonds", "10s": "10 of Spades", "10c": "10 of Clubs",
    "Jh": "Jack of Hearts", "Jd": "Jack of Diamonds", "Js": "Jack of Spades", "Jc": "Jack of Clubs",
    "Qh": "Queen of Hearts", "Qd": "Queen of Diamonds", "Qs": "Queen of Spades", "Qc": "Queen of Clubs",
    "Kh": "King of Hearts", "Kd": "King of Diamonds", "Ks": "King of Spades", "Kc": "King of Clubs"
}

# -------------------- HIGHLIGHT PERSISTENCE --------------------
last_box = None
highlight_frames_left = 0
last_target_card = "Jd"

# -------------------- ANNOTATION FUNCTION --------------------
def annotate_frame(frame, results, target_card):
    global last_box, highlight_frames_left, last_target_card
    annotated = frame.copy()
    overlay = annotated.copy()

    detected = False
    last_target_card = target_card

    if results and len(results) > 0:
        for box, cls_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            class_name = results[0].names[int(cls_id)]
            display_name = CARD_MAP.get(class_name, class_name)

            if class_name == target_card and conf > CONF_THRESHOLD:
                last_box = map(int, box)
                highlight_frames_left = PERSIST_FRAMES
                detected = True
                break  # only highlight one instance per frame

    # If not detected this frame, keep showing last highlight if counter > 0
    if not detected and highlight_frames_left > 0:
        highlight_frames_left -= 1

    # Draw the highlight if we have a valid box
    if last_box and highlight_frames_left > 0:
        x1, y1, x2, y2 = last_box

        # yellow fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, ALPHA_FILL, annotated, 1 - ALPHA_FILL, 0, annotated)

        # red border
        overlay_border = annotated.copy()
        cv2.rectangle(overlay_border, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.addWeighted(overlay_border, ALPHA_BORDER, annotated, 1 - ALPHA_BORDER, 0, annotated)

        # centered text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, _ = cv2.getTextSize(CARD_MAP.get(target_card, target_card), font, font_scale, thickness)
        text_width, text_height = text_size
        text_x = int(x1 + (x2 - x1 - text_width) / 2)
        text_y = y1 - 10
        if text_y - text_height < 0:
            text_y = y2 + text_height + 10

        overlay_text = annotated.copy()
        # outline
        cv2.putText(overlay_text, CARD_MAP.get(target_card, target_card), (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # main text
        cv2.putText(overlay_text, CARD_MAP.get(target_card, target_card), (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay_text, ALPHA_TEXT, annotated, 1 - ALPHA_TEXT, 0, annotated)

    return annotated

# -------------------- FLASK ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    target_card = request.form.get('target_card', last_target_card)
    file = request.files['frame']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    results = model(img)
    annotated = annotate_frame(img, results, target_card)

    _, buffer = cv2.imencode('.jpg', annotated)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')

# -------------------- RUN --------------------
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


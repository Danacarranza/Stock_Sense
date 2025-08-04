import cv2
import time
import json
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIGURACIÓN
# =========================
VIDEO_SOURCE = 1  # Cámara del iPhone
MODEL_PATH = "models/yolov8n.pt"
DETECTION_INTERVAL = 5  # segundos

# =========================
# INICIALIZAR MODELO
# =========================
model = YOLO(MODEL_PATH)

# =========================
# DIBUJAR BOUNDING BOXES
# =========================
def draw_boxes(image, results, model):
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 18)
    except:
        font = ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            text = f"{label} ({conf*100:.1f}%)"

            try:
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = font.getsize(text)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
            draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# =========================
# LOOP DE DETECCIÓN
# =========================
def detect_and_save():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            continue

        results = model(frame, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append(label)

        counts = dict(Counter(detections))
        print("📊 Conteo:", counts)

        with open("counts.json", "w") as f:
            json.dump(counts, f)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("timestamp.txt", "w") as f:
            f.write(timestamp)

        final_image = draw_boxes(frame, results, model)
        cv2.imwrite("last_frame.jpg", final_image)

        time.sleep(DETECTION_INTERVAL)

    cap.release()

# =========================
# EJECUCIÓN PRINCIPAL
# =========================
if __name__ == "__main__":
    detect_and_save()

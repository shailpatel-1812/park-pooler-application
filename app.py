from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import torch
import numpy as np
from paddleocr import PaddleOCR
import threading
import time

# Custom model path
model_path = r"best.pt"

# Choose device; "cpu" or "cuda"(if cuda is available)
cpu_or_cuda = "cpu"
device = torch.device(cpu_or_cuda)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = model.to(device)

# Load OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', max_text_length=16)

# Create the directory for storing license plate images
output_dir = "license_plates"
os.makedirs(output_dir, exist_ok=True)

app = Flask(__name__)

# Store detected texts globally
detected_texts = []
uploaded_files = {}
processing_status = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['file']
        filename = secure_filename(video_file.filename)
        video_file.save(os.path.join("temp", filename))
        uploaded_files[filename] = filename
        processing_status[filename] = "processing"

        thread = threading.Thread(target=process_video, args=(filename,))
        thread.start()

        return render_template('loading.html', filename=filename)

    return render_template('index.html')

@app.route('/status/<filename>', methods=['GET'])
def status(filename):
    return jsonify({"status": processing_status.get(filename, "processing")})

@app.route('/result/<filename>', methods=['GET'])
def result(filename):
    if processing_status.get(filename) == "done":
        return render_template('result.html', video_filename=filename, largest_box_texts=detected_texts)
    else:
        return "Processing not finished yet. Please wait.", 202

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('temp', filename)

def process_video(filename):
    video_path = os.path.join("temp", filename)
    frame = cv2.VideoCapture(video_path)
    frame_width = int(frame.get(3))
    frame_height = int(frame.get(4))
    size = (frame_width, frame_height)
    writer = cv2.VideoWriter('output.mp4',-1,8,size)
    text_font = cv2.FONT_HERSHEY_PLAIN
    color= (0,0,255)
    text_font_scale = 1.25
    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    seen_texts = set()
    while True:
        ret, image = frame.read()
        frame_count += 1
        if ret:
            output = model(image)
            result = np.array(output.pandas().xyxy[0])
            for i in result:
                p1 = (int(i[0]), int(i[1]))
                p2 = (int(i[2]), int(i[3]))
                text_origin = (int(i[0]), int(i[1]) - 5)
                if i[-3] > 0.85:
                    cv2.rectangle(image, p1, p2, color=color, thickness=2)
                    cv2.putText(image, text=f"{i[-1]} {i[-3]:.2f}", org=text_origin,
                                fontFace=text_font, fontScale=text_font_scale,
                                color=color, thickness=2)
                    box_image = image[p1[1]:p2[1], p1[0]:p2[0]]
                    result = ocr.ocr(box_image, cls=True)
                    for idx in range(len(result)):
                        res = result[idx]
                        max_area = 0
                        largest_box = None
                        for box in res:
                            polygon = box[0]
                            area = (polygon[2][0] - polygon[0][0]) * (polygon[2][1] - polygon[0][1])
                            if area > max_area:
                                max_area = area
                                largest_box = box
                        if largest_box[1][0] not in seen_texts:
                            seen_texts.add(largest_box[1][0])
                            detected_texts.append(largest_box[1][0])
                        output_path = os.path.join(output_dir, f"bounding_box_{frame_count}.jpg")
                        cv2.imwrite(output_path, box_image)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(image, fps, (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            writer.write(image)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame.release()
    cv2.destroyAllWindows()
    processing_status[filename] = "done"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')


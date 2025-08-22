from flask import Flask , request , render_template , Response , session
import os
import cv2 
import time
import math
from ultralytics import YOLO


app = Flask(__name__)     # Initialize Flask application
UPLOAD_FOULDER = 'uploads'
os.makedirs(UPLOAD_FOULDER,exist_ok=True)    # Create upload directory if it doesn't exist

model = YOLO('yolov12m.pt')
cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]


"""
Generator function that processes video frames and yields JPEG-encoded frames for streaming.
Args:
    video_path (str): Path to the video file to process    
Yields:
    bytes: JPEG-encoded frame data for HTTP streaming
"""
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    ptime = 0 

    while True:
        ret , frame = cap.read()
        if not ret :
            break
        
        results = model.predict(frame,conf = 0.15 , iou = 0.1)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1,y1,x2,y2 = map(int , box.xyxy[0])      # Extract bounding box coordinates
                cv2.rectangle(frame, (x1,y1),(x2,y2),[255,0,0],2)

                # Get class information
                class_id = int(box.cls[0])
                class_name = cocoClassNames[class_id]
                conf = round(box.conf[0].item() , 2)

                label = f'{class_name} : {conf}'
                text_size = cv2.getTextSize(label,0,0.5,2)[0]
                c2 = x1 + text_size[0] , y1 - text_size[1] - 3
                cv2.rectangle(frame,(x1,y1), c2,[255,0,255], -1)
                cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,[255,255,255],1)
        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(frame,f'FPS : {fps}',(30,70),cv2.FONT_HERSHEY_PLAIN,2,[255,0,255],2)

        # Encode frame as JPEG for streaming
        ret , buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
          
        # Yield frame in multipart format for HTTP streaming
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

"""
Handle file upload and render the main page.
Returns:
    rendered template: HTML page with upload form or video display
"""
@app.route('/', methods=['GET' , 'POST'])
def upload_file():
    if request.method == 'POST':        # Check if file was uploaded
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        file_path = os.path.join(UPLOAD_FOULDER,file.filename)
        file.save(file_path)

        # Render template with video path
        return render_template('index.html', video_path=file.filename)
    return render_template('index.html')


"""
Video streaming route that serves processed frames.
Args:
    video_filename (str): Name of the video file to process    
Returns:
    Response: HTTP response with multipart video stream
"""
@app.route('/video_feed/<video_filename>')
def video_feed(video_filename):
    video_path = os.path.join(UPLOAD_FOULDER,video_filename)
    return Response(generate_frames(video_path),mimetype='multipart/x-mixes-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)     # Run Flask application in debug mode
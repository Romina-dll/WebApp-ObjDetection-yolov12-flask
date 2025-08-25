from flask import Flask, render_template, Response, jsonify, request  # Flask web framework components
from werkzeug.utils import secure_filename  # Secure filename handling for uploads
import os  
import cv2  
import time 
import numpy as np
from ultralytics import YOLO  

# Initialize Flask Application
app = Flask(__name__)  # Create Flask application instance
app.config['UPLOAD_FOLDER'] = 'uploads'  # Configure upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload directory exists

# Load YOLO Model
model = YOLO("yolov12m.pt")  
cocoClassNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


"""
Render the main application interface.
Returns:
    Rendered HTML template: The main dashboard page
"""
@app.route('/')
def index():
    return render_template('index.html')

"""
Handle video file uploads from the client.
Processes multipart/form-data uploads, validates the file, and saves it to the server's upload directory.
Returns:
    JSON response: Success message or error with appropriate HTTP status code
"""
@app.route('/upload', methods=['POST'])
def upload_file():
    
    global uploaded_video_path

    # Validate file presence in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    # Validate filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Secure filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        uploaded_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file to server
        file.save(uploaded_video_path)
        print(f"Uploaded file: {uploaded_video_path}")  # Debugging log

        return jsonify({
            'message': 'File uploaded successfully!', 
            'video_path': f"/uploads/{filename}"
        })

"""
Video frame generator for real-time object detection.
Processes each frame of the uploaded video, performs object detection using YOLO, draws bounding boxes and labels, and yields JPEG-encoded frames for streaming.
Yields:
    bytes: JPEG-encoded video frames with detection annotations
"""
def generate_frames():
    
    global uploaded_video_path, latest_person_count

    # Validate video file existence
    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        print("Error: No video uploaded.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(uploaded_video_path)
    ptime = 0  # Previous time for FPS calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform object detection with confidence and IOU thresholds
        results = model.predict(frame, conf=0.15, iou=0.1)
        person_count = 0  # Initialize person counter for current frame

        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)
                
                # Get class information
                classNameInt = int(box.cls[0])
                classname = cocoClassNames[classNameInt]
                
                # Count persons specifically
                if classname == "person":
                    person_count += 1
                
                # Prepare and display label
                conf = round(box.conf[0].item(), 2)
                label = f"{classname}: {conf}"
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
        
        # Update global person count
        latest_person_count = person_count
        
        # Calculate and display FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Cleanup
    cap.release()


"""
Provide real-time video stream with object detection.
Returns:
    Response: Multipart HTTP response containing video stream
    JSON error: If no video is available for processing
"""
@app.route('/video_feed')
def video_feed():
    
    global uploaded_video_path
    
    # Validate video availability
    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        return jsonify({'error': 'No video uploaded'}), 400

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


"""
API endpoint to retrieve the latest person count.
Returns:
    JSON: Current person count from the most recently processed frame
"""
@app.route('/person_count')
def person_count():
    return jsonify({"count": latest_person_count}) #API to return the latest person count



if __name__ == "__main__":
    # Start Flask development server
    app.run(debug=True, threaded=True)
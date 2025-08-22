import cv2
import math
import time
from ultralytics import YOLO

"""
Detect objects in a video using YOLO model and save the output video with bounding boxes.
Args:
    video_path (str): Path to the input video file
    model_path (str): Path to the YOLO model weights file
    output_path (str): Path to save the output video with detections
    conf_thresh (float): Confidence threshold for detections (0-1)
    iou_thresh (float): IOU threshold for non-max suppression (0-1)
"""
def detect_object_yolov12(video_path , model_path = 'yolov12l.pt' , output_path = 'output.mp4' , conf_thresh = 0.15 , iou_thresh = 0.1):
    #load model
    model = YOLO(model_path)

    #Open video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    #Difine vieo writter
    output_video = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width,frame_height)
    )

    #coco class name 
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
    ptime = 0 
    count = 0 

    while True:
        ret , frame = cap.read()
        if not ret:
            break
        count += 1
        results = model.predict(frame,conf=conf_thresh , iou = iou_thresh)

        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1) , (x2,y2) ,[255,0,0],2)
                class_id = int(box.cls[0])
                class_name = cocoClassNames[class_id]
                conf = round(float(box.conf[0]),2)
                label = f'{class_name} : {conf}'  # Create label text
                # Calculate text background size
                text_size = cv2.getTextSize(label,0,fontScale=0.5,thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame,(x1,y1) , c2 , [255,0,0],-1)
                cv2.putText(frame,label,(x1,y1-2),0,0.5,[255,255,255],1,cv2.LINE_AA)
        
        #caculate and display FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
        ptime = ctime
        cv2.putText(frame,f'FPS: {int(fps)}' , (30,70) , cv2.FONT_HERSHEY_PLAIN , 3, (255,0,255) ,3)
        
        #Write Frame to output video
        output_video.write(frame)
        cv2.imshow('Video' , frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    
detect_object_yolov12('Resources/Videos/video2.mp4')



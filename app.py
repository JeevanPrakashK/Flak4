from flask import Flask, render_template, Response
from imutils.video import VideoStream
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    lower_green = np.array([35, 100, 100])  # Lower bounds (BGR)
    upper_green = np.array([77, 255, 255])  # Upper bounds (BGR)
    
    cap_video = cv2.VideoCapture("background_video.mp4")
    cap_webcam = VideoStream(src=0).start()  # Start video stream from default camera
    
    if not cap_video.isOpened() or not cap_webcam:
        print("Error opening video capture(s)")
        return
    
    while True:
        frame_video = cap_video.read()[1]
        frame_webcam = cap_webcam.read()
        
        hsv_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_video, lower_green, upper_green)
        inv_mask = cv2.bitwise_not(mask)
        foreground = cv2.bitwise_and(frame_video, frame_video, mask=inv_mask)
        
        if 'mask_size' not in globals():
            mask_size = (frame_webcam.shape[1], frame_webcam.shape[0])
            frame_size = (frame_webcam.shape[1], frame_webcam.shape[0])
        
        mask = cv2.resize(mask, mask_size)
        background = cv2.bitwise_and(frame_webcam, frame_webcam, mask=mask)
        foreground = cv2.resize(foreground, frame_size)
        background = cv2.resize(background, frame_size)
        final_frame = cv2.add(foreground, background)
        
        ret, buffer = cv2.imencode('.jpg', final_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap_video.release()
    cap_webcam.stop()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)

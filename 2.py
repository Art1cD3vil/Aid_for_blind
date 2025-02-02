import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import pyttsx3
import threading
from queue import Queue

# Load the YOLOv8 model with classification capabilities
model = YOLO("best_2.pt")

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Specify the IP address and port of the camera stream
camera_url = 0
cap = cv2.VideoCapture(camera_url)

# Check if the camera stream is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera stream.")
    exit()

# Set the video writer parameters if you want to save the output
w, h, fps = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

center_point = (w // 2, h)  # Center point of the frame
known_width = 0.5  # Known width of the object in meters (you need to set this)
focal_length = 1000  # Focal length in pixels (you need to calibrate this)

txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

text_font_scale = 0.5  # Adjust this value to change the text size
text_thickness = 2  # Adjust this value to change the text thickness

feedback_queue = Queue()
frame_queue = Queue(maxsize=10)

def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        if not frame_queue.full():
            frame_queue.put(frame)

def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            annotator = Annotator(frame, line_width=2)
            results = model(frame)

            for result in results:
                boxes = result.boxes.xyxy.cpu()

                for i, box in enumerate(boxes):
                    cls = int(result.boxes.cls[i])
                    label = model.names[cls]
                    annotator.box_label(box, label, color=colors(cls, True))
                
                    x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid
                    box_width = box[2] - box[0]
                    distance = (known_width * focal_length) / box_width

                    text = f"{distance:.2f} m"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)

                    # Adjust the text position based on the bounding box location
                    text_x = max(10, x1 - text_size[0] // 2)
                    text_x = min(text_x, frame.shape[1] - text_size[0] - 10)
                    text_y = max(y1 - text_size[1] - 10, 10)

                    cv2.rectangle(frame, (text_x - 5, text_y - 5), (text_x + text_size[0] + 5, text_y + text_size[1] + 5), txt_background, -1)
                    cv2.putText(frame, text, (text_x, text_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, txt_color, text_thickness)

                    # Check for pothole detection and provide voice guidance
                    if label == "stairs":
                        if x1 < w // 3:  # Pothole on the left
                            feedback_message = f"Pothole on the left, move right or walk straight. Distance: {distance:.2f} meters."
                        elif x1 > 2 * (w // 3):  # Pothole on the right
                            feedback_message = f"Pothole on the right, move left or walk straight. Distance: {distance:.2f} meters."
                        else:  # Pothole in the center
                            feedback_message = f"Pothole straight ahead, move left or right. Distance: {distance:.2f} meters."
                        
                        print(feedback_message)
                        feedback_queue.put(feedback_message)

            # Write the frame to the video writer (if enabled)
            out.write(frame)

            # Display the frame
            cv2.imshow("visioneye-distance-calculation", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def give_voice_feedback():
    while True:
        message = feedback_queue.get()
        if message is None:
            break
        engine.say(message)
        engine.runAndWait()

# Start threads for capturing and processing frames
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
tts_thread = threading.Thread(target=give_voice_feedback)

capture_thread.start()
process_thread.start()
tts_thread.start()

try:
    capture_thread.join()
    process_thread.join()
finally:
    # Stop the TTS thread
    feedback_queue.put(None)
    tts_thread.join()

    # Release resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()

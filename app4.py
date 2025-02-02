import cv2
import math
import time
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

feedback_queue = Queue()  # Create a Queue instance

def give_voice_feedback():
    last_message = None
    while True:
        message = feedback_queue.get()
        if message is None:
            break
        if message != last_message:
            engine.say(message)
            engine.runAndWait()
            last_message = message

tts_thread = threading.Thread(target=give_voice_feedback)
tts_thread.start()

# Desired frame rate (frames per second)
desired_fps = 2
frame_interval = 1.0 / desired_fps

last_feedback = None  # To track the last feedback message
feedback_cooldown = 2  # Cooldown period in seconds to prevent repeated feedback
last_feedback_time = 0

try:
    while True:
        # Start time for frame capture
        start_time = time.time()

        # Read the frame from the camera stream
        ret, im0 = cap.read()
        if not ret:
            print("Error capturing frame from the camera stream.")
            break

        # Capture time after frame read
        capture_time = time.time()

        annotator = Annotator(im0, line_width=2)
        results = model(im0)

        # Time after processing
        process_time = time.time()

        detected = False  # Flag to check if the object is detected

        for result in results:
            boxes = result.boxes.xyxy.cpu()

            for i, box in enumerate(boxes):
                cls = int(result.boxes.cls[i])
                label = model.names[cls]
                annotator.box_label(box, label, color=colors(cls, True))
                annotator.visioneye(box, center_point)
            
                x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid
                box_width = box[2] - box[0]
                distance = (known_width * focal_length) / box_width

                text = f"{distance:.2f} m"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)

                # Adjust the text position based on the bounding box location
                text_x = max(10, x1 - text_size[0] // 2)
                text_x = min(text_x, im0.shape[1] - text_size[0] - 10)
                text_y = max(y1 - text_size[1] - 10, 10)

                cv2.rectangle(im0, (text_x - 5, text_y - 5), (text_x + text_size[0] + 5, text_y + text_size[1] + 5), txt_background, -1)
                cv2.putText(im0, text, (text_x, text_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, txt_color, text_thickness)

                # Check for pothole detection and provide voice guidance
                if label == "Pothole":
                    detected = True
                    current_time = time.time()
                    if current_time - last_feedback_time >= feedback_cooldown:
                        if x1 < w // 3:  # Pothole on the left
                            feedback_message = f"Pothole on the left, move right or walk straight. Distance: {distance:.2f} meters."
                        elif x1 > 2 * (w // 3):  # Pothole on the right
                            feedback_message = f"Pothole on the right, move left or walk straight. Distance: {distance:.2f} meters."
                        else:  # Pothole in the center
                            feedback_message = f"Pothole straight ahead, move left or right. Distance: {distance:.2f} meters."

                        if feedback_message != last_feedback:
                            print(feedback_message)
                            feedback_queue.put(feedback_message)
                            last_feedback = feedback_message
                            last_feedback_time = current_time

        if not detected:
            # No object detected, reset feedback
            last_feedback = None

        # Write the frame to the video writer (if enabled)
        out.write(im0)

        # Display the frame
        cv2.imshow("visioneye-distance-calculation", im0)

        # Capture time after displaying
        display_time = time.time()

        # Calculate the time taken for processing
        frame_capture_time = capture_time - start_time
        frame_processing_time = process_time - capture_time
        frame_display_time = display_time - process_time

        print(f"Frame capture time: {frame_capture_time:.3f} seconds")
        print(f"Frame processing time: {frame_processing_time:.3f} seconds")
        print(f"Frame display time: {frame_display_time:.3f} seconds")

        # Calculate the time to wait to match the desired frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed_time)
        time.sleep(sleep_time)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the TTS thread
    feedback_queue.put(None)
    tts_thread.join()

    # Release resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()

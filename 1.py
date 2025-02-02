import cv2
import math
import pyttsx3
from ultralytics import YOLO 
from ultralytics.utils.plotting import Annotator, colors

# Load the YOLOv8 model with classification capabilities
model = YOLO("25_best.pt")

# Path to the input image
image_path = '1.jpg'

# Output path for the annotated image
output_path = 'output_image.jpg'

# Known width of the object in meters (you need to set this)
known_width = 0.5

# Focal length in pixels (you need to calibrate this)
focal_length = 700

txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

text_font_scale = 0.5  # Adjust this value to change the text size
text_thickness = 2  # Adjust this value to change the text thickness

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()

# Read the input image
im0 = cv2.imread(image_path)

if im0 is None:
    print(f"Error: Unable to load the image {image_path}.")
    exit()

h, w, _ = im0.shape
center_point = (w // 2, h)  # Center point of the frame

annotator = Annotator(im0, line_width=2)
results = model(im0)

for result in results:
    boxes = result.boxes.xyxy.cpu()

    for i, box in enumerate(boxes):
        cls = int(result.boxes.cls[i])
        label = model.names[cls]
        
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        box_width = x2 - x1
        distance = (known_width * focal_length) / box_width

        if distance <= 2.0:  # Only consider objects within 2 meters
            annotator.box_label(box, label, color=colors(cls, True))

            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2  # Bounding box centroid
            text = f"{distance:.2f} m"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)

            # Adjust the text position based on the bounding box location
            text_x = max(10, x_center - text_size[0] // 2)
            text_x = min(text_x, im0.shape[1] - text_size[0] - 10)
            text_y = max(y_center - text_size[1] - 10, 10)

            cv2.rectangle(im0, (text_x - 5, text_y - 5), (text_x + text_size[0] + 5, text_y + text_size[1] + 5), txt_background, -1)
            cv2.putText(im0, text, (text_x, text_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, txt_color, text_thickness)

            # Provide the distance in voice
            tts_engine.say(f"{label} detected at {distance:.2f} meters ahead")
            tts_engine.runAndWait()

# Save the output image
cv2.imwrite(output_path, im0)

# Display the image (optional)
cv2.imshow("visioneye-distance-calculation", im0)
# Press any key to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

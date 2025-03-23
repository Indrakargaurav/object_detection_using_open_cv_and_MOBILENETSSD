import numpy as np
import cv2

# File paths
image_path = 'check.jpg'
prototxt_path = 'MobileNetSSD_deploy.prototxt'
model_path = 'MobileNetSSD_deploy.caffemodel'

# Minimum confidence threshold
min_confidence = 0.6

# Object classes
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

# Generate random colors for classes
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the pre-trained Caffe model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the input image
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Preprocess the image for the network
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), scalefactor=0.007843, size=(300, 300), mean=127.5)

# Pass the blob through the network
net.setInput(blob)
detected_objects = net.forward()

# Process detected objects
for i in range(detected_objects.shape[2]):
    confidence = detected_objects[0, 0, i, 2]

    if confidence > min_confidence:
        class_index = int(detected_objects[0, 0, i, 1])

        # Get bounding box coordinates
        upper_left_x = int(detected_objects[0, 0, i, 3] * width)
        upper_left_y = int(detected_objects[0, 0, i, 4] * height)
        lower_right_x = int(detected_objects[0, 0, i, 5] * width)
        lower_right_y = int(detected_objects[0, 0, i, 6] * height)

        # Prediction text
        prediction_text = f"{classes[class_index]}: {confidence:.2f}%"

        # Draw bounding box and label
        color = colors[class_index % len(colors)]  # Prevent index out of range
        cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), color, 3)
        cv2.putText(image, prediction_text, (upper_left_x, upper_left_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Show the image with detected objects
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

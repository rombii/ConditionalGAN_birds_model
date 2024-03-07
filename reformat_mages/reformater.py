import cv2
import numpy as np
import glob
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

wanted_ids = [16, 14, 21]

# Get all image paths
image_paths = glob.glob('../Data/CUB_200_2011/images/*/*.jpg')
# image_paths = [image_paths[9]]
for image_path in image_paths:
    print("Processing", image_path)
    # Loading image
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2 and class_id in wanted_ids:
                # print(confidence, class_id)
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # Calculate the center of the bounding box
            center_x, center_y = x + w // 2, y + h // 2
            # Determine the maximum dimension
            max_dim = max(w, h)
            # Adjust the x and y coordinates to create a square bounding box
            x = center_x - max_dim // 2
            y = center_y - max_dim // 2
            # Ensure coordinates are within image dimensions
            x, y = max(0, x), max(0, y)
            w, h = min(img.shape[1] - x, max_dim), min(img.shape[0] - y, max_dim)
            # Crop the image
            cropped_image = img[y:y + h, x:x + w]
            # Check if the cropped image is not empty
            if cropped_image.size > 0:
                # Resize the cropped image to the desired size
                final_image = cv2.resize(cropped_image, (64, 64))

                # Create the new directory if it doesn't exist
                new_dir = image_path.replace('Data/CUB_200_2011/images', 'Data/CUB_200_2011_reformat/images')
                os.makedirs(os.path.dirname(new_dir), exist_ok=True)
                # Save the image
                cv2.imwrite(new_dir, final_image)
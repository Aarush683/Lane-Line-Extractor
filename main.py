
import cv2
import numpy as np
import os

# ==========================================
# CLASSICAL COMPUTER VISION
# ROAD EDGE DETECTION USING ROI
# ==========================================


# Step 1: Load the road image

image_path = "images/road.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load images/road.jpg")
    print("Check your folder and image filename.")
    raise SystemExit

height, width = image.shape[:2]

print("Image loaded successfully!")
print("Image dimensions:", width, "x", height)


# Step 2: Convert BGR to HSV

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Step 3: Define yellow and white colour ranges

# Yellow lane markings

yellow_lower = np.array([15, 70, 70])
yellow_upper = np.array([40, 255, 255])

# White lane markings

white_lower = np.array([0, 0, 180])
white_upper = np.array([180, 70, 255])


# Step 4: Create colour masks

yellow_mask = cv2.inRange(
    hsv, yellow_lower, yellow_upper
)

white_mask = cv2.inRange(
    hsv, white_lower, white_upper
)

# Combine yellow and white masks

color_mask = cv2.bitwise_or(
    yellow_mask, white_mask
)


# Step 5: Remove small noise using morphological opening

kernel = np.ones((3, 3), np.uint8)

color_mask = cv2.morphologyEx(
    color_mask, cv2.MORPH_OPEN, kernel
)


# Step 6: Keep only selected colour pixels

filtered_image = cv2.bitwise_and(
    image, image, mask=color_mask
)


# Step 7: Convert the filtered image to grayscale

gray = cv2.cvtColor(
    filtered_image, cv2.COLOR_BGR2GRAY
)


# Step 8: Apply Gaussian blur

blurred = cv2.GaussianBlur(
    gray, (5, 5), 0
)


# Step 9: Detect edges using Canny

edges = cv2.Canny(
    blurred, 40, 120
)


# Step 10: Define the Region of Interest (ROI)

roi_vertices = np.array([[
    (int(width * 0.18), height - 1),
    (int(width * 0.78), height - 1),
    (int(width * 0.60), int(height * 0.48)),
    (int(width * 0.40), int(height * 0.48))
]], dtype=np.int32)


# Step 11: Create the ROI mask

roi_mask = np.zeros_like(edges)

cv2.fillPoly(
    roi_mask, roi_vertices, 255
)


# Step 12: Apply the ROI mask to the edge image

roi_edges = cv2.bitwise_and(
    edges, roi_mask
)


# Step 13: Save the processing results

output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

outputs = {
    "color_mask.jpg": color_mask,
    "filtered_road.jpg": filtered_image,
    "gaussian_blur.jpg": blurred,
    "canny_edges.jpg": edges,
    "roi_edges.jpg": roi_edges
}

for filename, result in outputs.items():
    filepath = os.path.join(output_folder, filename)

    if not cv2.imwrite(filepath, result):
        print("Could not save:", filepath)

print("Processing completed!")
print("Results saved in the output folder.")


# Step 14: Display the processing stages

cv2.imshow("Original Road Image", image)

cv2.imshow("HSV Colour Mask", color_mask)

cv2.imshow("Filtered Road", filtered_image)

cv2.imshow("Gaussian Blur", blurred)

cv2.imshow("Canny Edges", edges)

cv2.imshow("ROI Edges - Final Output", roi_edges)

cv2.waitKey(0)

cv2.destroyAllWindows()








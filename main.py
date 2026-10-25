
import cv2
import numpy as np
import os
import math

# ==========================================
# CLASSICAL COMPUTER VISION
# CURVED ROAD LANE LINE EXTRACTOR
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

# Yellow lane markings
yellow_lower = np.array([15, 70, 70])
yellow_upper = np.array([40, 255, 255])

# White lane markings
white_lower = np.array([0, 0, 180])
white_upper = np.array([180, 70, 255])

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

# Step 3: Clean the color mask slightly
kernel = np.ones((3, 3), np.uint8)

color_mask = cv2.morphologyEx(
    color_mask, cv2.MORPH_OPEN, kernel
)

# Keep only selected color pixels
filtered_image = cv2.bitwise_and(
    image, image, mask=color_mask
)

# Step 4: Convert the filtered image to grayscale
gray = cv2.cvtColor(
    filtered_image, cv2.COLOR_BGR2GRAY
)

# Step 5: Apply Gaussian blur
blurred = cv2.GaussianBlur(
    gray, (5, 5), 0
)

# Step 6: Detect edges using Canny
edges = cv2.Canny(
    blurred, 40, 120
)

# Step 7: Define a narrower Region of Interest
roi_vertices = np.array([[
    (int(width * 0.18), height - 1),
    (int(width * 0.78), height - 1),
    (int(width * 0.60), int(height * 0.48)),
    (int(width * 0.40), int(height * 0.48))
]], dtype=np.int32)

roi_mask = np.zeros_like(edges)

cv2.fillPoly(
    roi_mask, roi_vertices, 255
)

roi_edges = cv2.bitwise_and(
    edges, roi_mask
)

# Step 8: Detect line segments
lines = cv2.HoughLinesP(
    roi_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=35,
    minLineLength=max(30, int(height * 0.06)),
    maxLineGap=max(15, int(height * 0.04))
)

# Step 9: Draw only plausible lane segments
line_overlay = image.copy()

left_lines = []
right_lines = []

if lines is not None:
    for line in lines:
        coords = np.asarray(line).reshape(-1)

        if coords.size != 4:
            continue

        x1, y1, x2, y2 = map(int, coords)

        dx = x2 - x1
        dy = y2 - y1

        # Ignore nearly horizontal segments
        if abs(dx) < 1:
            continue

        slope = dy / dx

        # Ignore very flat segments
        if abs(slope) < 0.25:
            continue

        midpoint_x = (x1 + x2) // 2

        # Classify segments by image midpoint.
        # This is a simple heuristic, not full curve tracking.
        if midpoint_x < width // 2:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

# Draw the accepted line segments
for x1, y1, x2, y2 in left_lines:
    cv2.line(
        line_overlay,
        (x1, y1), (x2, y2),
        (0, 255, 0), 3
    )

for x1, y1, x2, y2 in right_lines:
    cv2.line(
        line_overlay,
        (x1, y1), (x2, y2),
        (0, 255, 0), 3
    )

print("Left-side segments:", len(left_lines))
print("Right-side segments:", len(right_lines))

# Step 10: Blend the detected segments with the original
final_output = cv2.addWeighted(
    image, 0.9,
    line_overlay, 0.6,
    0
)

# Step 11: Save all results
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

outputs = {
    "color_mask.jpg": color_mask,
    "filtered_road.jpg": filtered_image,
    "gaussian_blur.jpg": blurred,
    "canny_edges.jpg": edges,
    "roi_edges.jpg": roi_edges,
    "lane_detection.jpg": final_output
}

for filename, result in outputs.items():
    filepath = os.path.join(output_folder, filename)

    if not cv2.imwrite(filepath, result):
        print("Could not save:", filepath)

print("Processing completed!")
print("Results saved in the output folder.")

# Step 12: Display the processing stages
cv2.imshow("Original Road Image", image)
cv2.imshow("HSV Color Mask", color_mask)
cv2.imshow("Filtered Road", filtered_image)
cv2.imshow("Gaussian Blur", blurred)
cv2.imshow("Canny Edges", edges)
cv2.imshow("ROI Edges", roi_edges)
cv2.imshow("Final Lane Detection", final_output)

cv2.waitKey(0)
cv2.destroyAllWindows()







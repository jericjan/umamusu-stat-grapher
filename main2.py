import cv2
import numpy as np
from PIL import Image, ImageGrab

# Grab image from clipboard
clipboard_image = ImageGrab.grabclipboard()
if clipboard_image is None:
    raise ValueError("No image found in clipboard!")

# Convert PIL image to OpenCV format
img_cv = cv2.cvtColor(np.array(clipboard_image), cv2.COLOR_RGB2BGR)


# Load the template (the cropped stat bar you provided)
template = cv2.imread("template.png")

# Convert both to grayscale
full_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform template matching
res = cv2.matchTemplate(full_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Get the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Get dimensions of the template
h, w = template_gray.shape

# Compute top-left and bottom-right points
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Crop from the full image
cropped = img_cv[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# Save or show
cv2.imwrite("cropped_stat_bar.png", cropped)

# Example approximate bounding boxes per stat
# Adjust these numbers to fit your stat bar image exactly!
boxes = [
    (41, 27, 59, 26),    # Speed
    (137, 26, 57, 27),
    (224, 27, 64, 26),
    (320, 26, 63, 27),
    (422, 27, 56, 26)
]

# Loop and save or process each box
for i, (x, y, bw, bh) in enumerate(boxes):
    stat = cropped[y:y+bh, x:x+bw]
    cv2.imwrite(f"stat_{i}.png", stat)

# cv2.imshow("Cropped", cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

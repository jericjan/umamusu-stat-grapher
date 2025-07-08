import cv2
import numpy as np
from PIL import Image, ImageGrab
import pytesseract
from io import BytesIO
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Grab image from clipboard
clipboard_image = ImageGrab.grabclipboard()
if clipboard_image is None:
    raise ValueError("No image found in clipboard!")

# Convert PIL image to OpenCV format
img_cv = cv2.cvtColor(np.array(clipboard_image), cv2.COLOR_RGB2BGR)

# Load the template (must be provided as "template.png")
template = cv2.imread("template.png")
if template is None:
    raise ValueError("Template image not found!")

# Convert both to grayscale
full_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform template matching
res = cv2.matchTemplate(full_gray, template_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Get dimensions of the template
h, w = template_gray.shape

# Compute top-left and bottom-right points
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Crop the stat bar
cropped = img_cv[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# Pre-defined bounding boxes for each stat
boxes = [
    (41, 27, 59, 26),    # Speed
    (137, 26, 57, 27),   # Stamina
    (224, 27, 64, 26),   # Power
    (320, 26, 63, 27),   # Guts
    (422, 27, 56, 26)    # Wit
]

stats = []
labels = ["Speed", "Stamina", "Power", "Guts", "Wit"]

for i, (x, y, bw, bh) in enumerate(boxes):
    stat_crop = cropped[y:y+bh, x:x+bw]

    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(stat_crop, cv2.COLOR_BGR2RGB))

    # Save to BytesIO instead of disk
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    img_for_ocr = Image.open(buffer)

    # OCR config
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(img_for_ocr, config=config).strip()

    try:
        value = int(text)
    except ValueError:
        value = 0

    stats.append(value)

# Plot using matplotlib
plt.figure(figsize=(8, 4))
bars = plt.bar(labels, stats, color="skyblue")
plt.title("Stat Overview")
plt.ylabel("Value")

# Add value labels on top of each bar
for bar, stat in zip(bars, stats):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(stat),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.show()

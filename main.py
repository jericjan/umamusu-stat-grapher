import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageGrab
from io import BytesIO

# Set tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Grab image from clipboard
clipboard_image = ImageGrab.grabclipboard()
if clipboard_image is None:
    raise ValueError("No image found in clipboard!")

# Convert PIL image to OpenCV format
img_cv = cv2.cvtColor(np.array(clipboard_image), cv2.COLOR_RGB2BGR)

# Convert to grayscale
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_TOZERO)
cv2.imwrite("processed.png", thresh)

# Convert back to PIL for pytesseract
thresh_pil = Image.fromarray(thresh)

# Save to BytesIO instead of disk
buffer = BytesIO()
thresh_pil.save(buffer, format="PNG")
buffer.seek(0)

# Re-open from BytesIO
image_for_ocr = Image.open(buffer)

# OCR config
config = "--psm 6 -c tessedit_char_whitelist=0123456789"

# Run OCR
text = pytesseract.image_to_string(image_for_ocr, config=config)

# Print raw OCR result
print("Raw OCR result:", repr(text))

# Post-process
lines = [line.strip() for line in text.splitlines() if line.strip()]

print("Parsed lines:", lines)

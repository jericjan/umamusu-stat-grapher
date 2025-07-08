import cv2
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("your_image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# You might need to tune these parameters
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("processed.png", thresh)

# Load the processed image
image = Image.open("processed.png")

# Optional: If the image is clean, you might not need to specify config, but it helps!
config = "--psm 6 -c tessedit_char_whitelist=0123456789"

# Run OCR
text = pytesseract.image_to_string(image, config=config)

# Print raw result
print("Raw OCR result:", repr(text))

# Post-process (optional): remove whitespace, split into lines
lines = [line.strip() for line in text.splitlines() if line.strip()]

print("Parsed lines:", lines)

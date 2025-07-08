import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import win32gui
import win32process
import win32con
import psutil
import mss

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

process_name = "UmamusumePrettyDerby.exe"  # <-- Change this to your game process name

# --- Find window handle by process name ---
def get_hwnd_by_process_name(proc_name):
    hwnds = []

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                exe = psutil.Process(pid).name()
                if exe.lower() == proc_name.lower():
                    hwnds.append(hwnd)
            except Exception:
                pass

    win32gui.EnumWindows(enum_handler, None)
    return hwnds[0] if hwnds else None

hwnd = get_hwnd_by_process_name(process_name)
if hwnd is None:
    raise Exception(f"Window for process '{process_name}' not found!")

# Get window rect
left, top, right, bottom = win32gui.GetWindowRect(hwnd)
width = right - left
height = bottom - top

# --- Take screenshot of window ---
with mss.mss() as sct:
    monitor = {"left": left, "top": top, "width": width, "height": height}
    sct_img = sct.grab(monitor)
    img_cv = np.array(sct_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)  # Convert to BGR

# --- Load template ---
template = cv2.imread("template.png")
if template is None:
    raise ValueError("Template image not found!")

full_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Template matching
res = cv2.matchTemplate(full_gray, template_gray, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)

h, w = template_gray.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cropped = img_cv[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# --- Define stat bounding boxes ---
boxes = [
    (41, 27, 59, 26),    # Speed
    (137, 26, 57, 27),   # Stamina
    (224, 27, 64, 26),   # Power
    (320, 26, 63, 27),   # Guts
    (422, 27, 56, 26)    # Wit
]
labels = ["Speed", "Stamina", "Power", "Guts", "Wit"]
stats = []

for i, (x, y, bw, bh) in enumerate(boxes):
    stat_crop = cropped[y:y+bh, x:x+bw]
    pil_img = Image.fromarray(cv2.cvtColor(stat_crop, cv2.COLOR_BGR2RGB))

    # BytesIO buffer
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    img_for_ocr = Image.open(buffer)

    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(img_for_ocr, config=config).strip()

    try:
        value = int(text)
    except ValueError:
        value = 0
    stats.append(value)

# --- Plot ---
plt.figure(figsize=(8, 4))
bars = plt.bar(labels, stats, color="skyblue")
plt.title("Stat Overview")
plt.ylabel("Value")

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

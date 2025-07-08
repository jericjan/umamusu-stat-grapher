import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import win32gui
import win32process
import psutil
import mss
import time
from concurrent.futures import ThreadPoolExecutor

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

process_name = "UmamusumePrettyDerby.exe"

def ocr_single_stat(image_array):

    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_TOZERO)
    # Convert back to PIL for pytesseract
    pil_img = Image.fromarray(thresh)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    img_for_ocr = Image.open(buffer)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(img_for_ocr, config=config).strip()
    try:
        value = int(text)
    except ValueError:
        value = 0
    return value

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

# Load template
template = cv2.imread("template.png")
if template is None:
    raise ValueError("Template image not found!")

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
h, w = template_gray.shape

boxes = [
    (41, 27, 59, 26),    # Speed
    (137, 26, 57, 27),   # Stamina
    (224, 27, 64, 26),   # Power
    (320, 26, 63, 27),   # Guts
    (422, 27, 56, 26)    # Wit
]
labels = ["Speed", "Stamina", "Power", "Guts", "Wit"]

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
ax2 = ax.twinx()  # Second y-axis

saved_stats = [0, 0, 0, 0, 0]

# Start loop
with mss.mss() as sct:
    while True:
        try:
            loop_start = time.perf_counter()

            # --- Screenshot
            t0 = time.perf_counter()
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top

            monitor = {"left": left, "top": top, "width": width, "height": height}
            sct_img = sct.grab(monitor)
            img_cv = np.array(sct_img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
            t1 = time.perf_counter()

            # --- Template match
            full_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(full_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)            
            if max_val < 0.7:
                plt.pause(1.0)
                continue
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cropped = img_cv[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            t2 = time.perf_counter()

            # --- OCR
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i, (x, y, bw, bh) in enumerate(boxes):
                    stat_crop = cropped[y:y+bh, x:x+bw]
                    futures.append(executor.submit(ocr_single_stat, stat_crop))

                stats = [f.result() for f in futures]
                
                
            t3 = time.perf_counter()

            if any(x == 0 for x in stats) or stats == saved_stats:
                plt.pause(1.0)
                continue

            saved_stats = stats

            # --- Plot
            stats_no_guts = stats.copy()
            stats_no_guts[3] = 0

            ax.clear()
            ax2.clear()

            bars1 = ax.bar(labels, stats, color="skyblue", label="Original")
            bars2 = ax2.bar(labels, stats_no_guts, color="orange", alpha=0.5, label="Without Guts")

            ax.set_title("Umamusume Stats Graph")
            ax.set_ylabel("Original Stats")
            ax2.set_ylabel("No Guts Stats")

            non_guts_indices = [0, 1, 2, 4]
            non_guts_values = [stats[i] for i in non_guts_indices]
            min_non_guts_value = min(non_guts_values)
            ax2.set_ylim(bottom=min_non_guts_value)

            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")

            for bar, stat in zip(bars1, stats):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(stat),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            for bar, stat in zip(bars2, stats_no_guts):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(stat),
                         ha='center', va='bottom', fontsize=8)

            plt.pause(0.1)
            t4 = time.perf_counter()

            loop_end = time.perf_counter()

            # --- Print timings
            print(f"Loop total: {(loop_end - loop_start)*1000:.1f} ms | "
                  f"Screenshot: {(t1 - t0)*1000:.1f} ms | "
                  f"Template: {(t2 - t1)*1000:.1f} ms | "
                  f"OCR: {(t3 - t2)*1000:.1f} ms | "
                  f"Plot: {(t4 - t3)*1000:.1f} ms")

        except KeyboardInterrupt:
            print("Stopped by user.")
            break

plt.ioff()
plt.show()

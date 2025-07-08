import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import win32gui
import win32process
import psutil
import mss
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv
import keyboard  # Import the keyboard library

load_dotenv()

# --- Gemini API Configuration ---
# IMPORTANT: Set your GOOGLE_API_KEY environment variable before running.
# You can get a free API key from Google AI Studio: https://aistudio.google.com/
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise Exception("GOOGLE_API_KEY environment variable not set. Please get a key and set it.")

# Initialize the Gemini model
# Using 'gemini-1.5-flash' is recommended for speed and cost-effectiveness.
vision_model = genai.GenerativeModel('gemma-3-27b-it')
# -----------------------------

process_name = "UmamusumePrettyDerby.exe"

def get_stats_with_gemini(image_array):
    """
    Sends an image to the Gemini API and extracts the stat values.
    """
    # Convert the OpenCV image (NumPy array) to a PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

    prompt = """
    You are an expert OCR system for the game Umamusume Pretty Derby.
    Analyze this image of the stats bar.
    Extract the five numerical stat values in this exact order: Speed, Stamina, Power, Guts, Wit.
    Provide ONLY a comma-separated list of the five numbers.
    Example output: 157,91,121,100,125
    """

    try:
        # Generate content using the model
        response = vision_model.generate_content([prompt, pil_img])
        
        # Clean up the response and parse it
        stats_text = response.text.strip()
        stats = [int(s.strip()) for s in stats_text.split(',')]

        # Validate that we got exactly 5 stats
        if len(stats) != 5:
            print(f"Error: Gemini returned {len(stats)} values, expected 5. Response: '{stats_text}'")
            return [0, 0, 0, 0, 0] # Return a default value on error

        return stats

    except Exception as e:
        print(f"An error occurred during the Gemini API call: {e}")
        # In case of API error or parsing failure, return a default/failure value
        return [0, 0, 0, 0, 0]


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

            # Check for numpad 0 key press
            if keyboard.is_pressed('0'):
                print("Numpad 0 pressed. Triggering Gemini call.")

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
                print("Template match!")
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cropped = img_cv[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                t2 = time.perf_counter()

                # --- Get stats using Gemini Vision API ---
                stats = get_stats_with_gemini(cropped)
                print("Gemini call done")
                print(stats)
                t3 = time.perf_counter()
                # ----------------------------------------

                if stats == saved_stats or sum(stats) == 0:
                    plt.pause(1.0)
                    continue
                print("stat changed")
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
                min_non_guts_value = min(non_guts_values) if non_guts_values else 0
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
                      f"Gemini: {(t3 - t2)*1000:.1f} ms | " # Changed label from OCR
                      f"Plot: {(t4 - t3)*1000:.1f} ms")
            else:
                plt.pause(0.1)  # Small pause if the key isn't pressed

        except KeyboardInterrupt:
            print("Stopped by user.")
            break

plt.ioff()
plt.show()
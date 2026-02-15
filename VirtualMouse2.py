from random import random
import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# ---------------- Global Variables ----------------
prev_x, prev_y = 0, 0
smoothening = 5
gesture_state = "NONE"
last_action_time = 0
action_delay = 0.4  # seconds

mouse = Controller()
screenWidth, screenHeight = pyautogui.size()

# ---------------- Mediapipe Setup ----------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1  # single hand for dominant-hand gestures
)

# ---------------- Gesture Helper Functions ----------------
def is_index_bent(landmarks):
    return util.get_angle(landmarks[5], landmarks[6], landmarks[8]) < 70

def is_middle_bent(landmarks):
    return util.get_angle(landmarks[9], landmarks[10], landmarks[12]) < 70

def is_thumb_extended(landmarks):
    return util.get_angle(landmarks[2], landmarks[3], landmarks[4]) > 120

def is_pinch(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = util.get_distance([thumb_tip, index_tip])

    # Ensure other fingers (middle, ring, pinky) are mostly extended
    middle_angle = util.get_angle(landmarks[9], landmarks[10], landmarks[12])
    ring_angle = util.get_angle(landmarks[13], landmarks[14], landmarks[16])
    pinky_angle = util.get_angle(landmarks[17], landmarks[18], landmarks[20])

    other_fingers_extended = middle_angle > 80 and ring_angle > 80 and pinky_angle > 80

    return distance < 0.08 and other_fingers_extended


def is_fist(landmarks):
    return (
        is_index_bent(landmarks) and
        is_middle_bent(landmarks) and
        util.get_angle(landmarks[13], landmarks[14], landmarks[16]) < 60
    )

# ---------------- Mouse Movement ----------------
def move_mouse(index_tip):
    global prev_x, prev_y

    x = int(index_tip.x * screenWidth)
    y = int(index_tip.y * screenHeight)

    # Deadzone
    if abs(x - prev_x) < 5 and abs(y - prev_y) < 5:
        return

    # Smooth movement
    curr_x = prev_x + (x - prev_x) / smoothening
    curr_y = prev_y + (y - prev_y) / smoothening

    pyautogui.moveTo(curr_x, curr_y, duration=0)
    prev_x, prev_y = curr_x, curr_y

# ---------------- Gesture Detection ----------------
def detect_gestures(frame, hand_landmarks):
    global gesture_state, last_action_time

    landmarks_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    current_time = time.time()

    # Determine Gesture
    index_bent = is_index_bent(landmarks_list)
    middle_bent = is_middle_bent(landmarks_list)
    thumb_extended = is_thumb_extended(landmarks_list)
    pinch = is_pinch(landmarks_list)
    fist = is_fist(landmarks_list)

    if fist:
        new_state = "SCREENSHOT"
    elif index_bent and middle_bent and thumb_extended:
        new_state = "DOUBLE_CLICK"
    elif pinch:
        new_state = "RIGHT_CLICK"
    elif index_bent and thumb_extended:
        new_state = "LEFT_CLICK"
    else:
        new_state = "MOVE"

    # Execute Gesture
    if new_state == "MOVE":
        move_mouse(index_tip)
        gesture_state = "MOVE"
    else:
        if new_state != gesture_state and current_time - last_action_time > action_delay:
            gesture_state = new_state
            last_action_time = current_time

            if new_state == "LEFT_CLICK":
                mouse.click(Button.left)
                cv2.putText(frame, "Left Click", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("LEFT CLICK")

            elif new_state == "RIGHT_CLICK":
                mouse.click(Button.right)
                cv2.putText(frame, "Right Click", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("RIGHT CLICK")

            elif new_state == "DOUBLE_CLICK":
                mouse.click(Button.left, 2)
                cv2.putText(frame, "Double Click", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                print("DOUBLE CLICK")

            elif new_state == "SCREENSHOT":
                im1 = pyautogui.screenshot()
                label = random.randint(1, 1000)
                im1.save(f'screenshot_{label}.png')
                cv2.putText(frame, f'Screenshot {label}', (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                print("SCREENSHOT")

# ------------------ TKINTER UI ------------------
root = tk.Tk()
root.title("Gesture-Based Virtual Mouse")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

title = tk.Label(root, text="Gesture-Based Virtual Mouse",
                 font=("Helvetica", 20, "bold"), fg="white", bg="#1e1e1e")
title.pack(pady=10)

video_label = tk.Label(root, width=640, height=480, bg="black")
video_label.pack(pady=20)

instruction = tk.Label(root, text="Single Hand Gestures: Index = Left Click | Pinch = Right Click | Fist = Screenshot | Index+Middle = Double Click",
                       font=("Helvetica", 12), fg="lightgray", bg="#1e1e1e")
instruction.pack(pady=10)

running = True

# ---------------- Camera Loop ----------------
def start_camera():
    global running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            detect_gestures(frame, hand_landmarks)

        # Tkinter frame update
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        time.sleep(0.005)  # reduce CPU usage

    cap.release()

def on_close():
    global running
    running = False
    root.destroy()

camera_thread = threading.Thread(target=start_camera)
camera_thread.daemon = True
camera_thread.start()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

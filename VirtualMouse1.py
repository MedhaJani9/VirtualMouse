import cv2
import mediapipe as mp
import pyautogui
import random
import util
import time
import os
from pynput.mouse import Button, Controller

last_screenshot_time = 0
mouse = Controller()
screenWidth, screenHeight = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

draw = mp.solutions.drawing_utils


# ------------------ UTIL FUNCTIONS ------------------

def find_index_finger_tip(hand_landmarks):
    return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]


def is_palm_open(landmarks_list):
    if len(landmarks_list) < 21:
        return False

    return (
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 160 and
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 160 and
        util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) > 160 and
        util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20]) > 160
    )


def move_mouse(index_finger_tip):
    x = int(index_finger_tip.x * screenWidth)
    y = int(index_finger_tip.y * screenHeight)
    pyautogui.moveTo(x, y)


# ------------------ GESTURE CHECKS ------------------

def is_left_click(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 90 and
        thumb_index_dist > 50
    )


def is_right_click(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 and
        thumb_index_dist > 50
    )


def is_double_click(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        thumb_index_dist > 50
    )


def is_screenshot(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
        thumb_index_dist < 50
    )


# ------------------ MAIN GESTURE LOGIC ------------------

def detect_gestures(frame, hand_landmarks, hand_label):
    global last_screenshot_time 
    landmarks_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    thumb_index_dist = util.get_distance([landmarks_list[4], landmarks_list[5]])

    # ✋ PALM OPEN → PAUSE
    if is_palm_open(landmarks_list):
        cv2.putText(frame, "Mouse Paused (Palm Open)", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        return

    index_finger_tip = find_index_finger_tip(hand_landmarks)

    # MOVE
    if thumb_index_dist < 50:
        move_mouse(index_finger_tip)

    # LEFT CLICK → LEFT HAND
    elif hand_label == "Left" and is_left_click(landmarks_list, thumb_index_dist):
        mouse.click(Button.left)
        cv2.putText(frame, "Left Click", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # RIGHT CLICK → RIGHT HAND
    elif hand_label == "Right" and is_right_click(landmarks_list, thumb_index_dist):
        mouse.click(Button.right)
        cv2.putText(frame, "Right Click", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # DOUBLE CLICK
    elif is_double_click(landmarks_list, thumb_index_dist):
        pyautogui.doubleClick()
        cv2.putText(frame, "Double Click", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # SCREENSHOT
    if is_screenshot(landmarks_list, thumb_index_dist):
        current_time = time.time()
        if current_time - last_screenshot_time > 1:
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            path = os.path.join(os.getcwd(), f'my_screenshot_{label}.png')
            im1.save(path)
            cv2.putText(frame, f'Screenshot Taken: {label}', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            last_screenshot_time = current_time


# ------------------ MAIN LOOP ------------------

def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        if processed.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                processed.multi_hand_landmarks,
                processed.multi_handedness
            ):
                hand_label = hand_info.classification[0].label  # "Left" or "Right"

                draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mpHands.HAND_CONNECTIONS
                )

                detect_gestures(frame, hand_landmarks, hand_label)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(1)
    draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            if processed.multi_hand_landmarks and processed.multi_handedness:
                for hand_landmarks, handedness in zip(
                        processed.multi_hand_landmarks,
                        processed.multi_handedness):

                    hand_label = handedness.classification[0].label  # "Left" or "Right"

                    draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mpHands.HAND_CONNECTIONS
                    )

                    detect_gestures(frame, hand_landmarks, hand_label)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
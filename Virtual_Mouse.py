from random import random
import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
mouse = Controller()
    
print(mp.__file__)
print(mp.__version__)

screenWidth, screenHeight = pyautogui.size()
mouse = Controller()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]  # Index finger tip
    
    return None


def detect_gestures(frame, landmarks_list, processed):
    if len(landmarks_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        # print(f'Index Finger Tip Coordinates: x={index_finger_tip.x}, y={index_finger_tip.y}')
        thumb_index_dist = util.get_distance([landmarks_list[4], landmarks_list[5]])

        if thumb_index_dist < 50 and util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90:
            move_mouse(index_finger_tip)

        #LEFT CLICK
        elif is_left_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        #RIGHT CLICK
        
        elif is_right_click(landmarks_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #DOUBLE CLICK
        elif is_double_click(landmarks_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        #SCREENSHOT    
        elif is_screenshot(landmarks_list, thumb_index_dist):
            # pyautogui.screenshot('screenshot.png')
            # cv2.putText(frame, 'Screenshot Taken', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, f'Screenshot Taken: my_screenshot_{label}.png', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


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

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screenWidth)
        y = int(index_finger_tip.y * screenHeight)
        pyautogui.moveTo(x, y)


def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils
    #reading frame by frame
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            #mirror the frame 
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmarks_list = []
            if processed.multi_hand_landmarks:
                hands_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hands_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hands_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))
            
            detect_gestures(frame, landmarks_list, processed)

                # print(landmarks_list)

                # for hand_landmarks in processed.multi_hand_landmarks:
                #     for id, lm in enumerate(hand_landmarks.landmark):
                #         h, w, c = frame.shape
                #         cx, cy = int(lm.x * w), int(lm.y * h)
                #         landmarks_list.append((id, cx, cy))
                    
                #     # Example: Print the coordinates of the index finger tip (landmark 8)
                #     for landmark in landmarks_list:
                #         if landmark[0] == 8:  # Index finger tip
                #             print(f'Index Finger Tip Coordinates: x={landmark[1]}, y={landmark[2]}')


            # if processed.multi_hand_landmarks:
            #     for hand_landmarks in processed.multi_hand_landmarks:
            #         draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


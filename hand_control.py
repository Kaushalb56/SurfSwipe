import cv2
import mediapipe as mp
import pyautogui
import time

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Cooldowns and gesture state
cooldown = {"jump": 1.0, "slide": 1.0, "lr": 0.5, "touch": 1.0}
last_action = {"jump": 0, "slide": 0, "lr": 0, "touch": 0}
gesture_reset_required = False
current_gesture = None

def fingers_up(handLms):
    lm = handLms.landmark
    fingers = []

    # Dynamic hand width for thumb threshold
    hand_width = abs(lm[17].x - lm[5].x)
    thumb_threshold = hand_width * 0.25  # Adjust sensitivity here (0.25 works well)

    thumb_open = abs(lm[4].x - lm[3].x) > thumb_threshold and lm[4].x > lm[2].x
    fingers.append(1 if thumb_open else 0)

    # Other fingers: compare tip to pip joint
    tips_ids = [8, 12, 16, 20]
    for tip_id in tips_ids:
        fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)

    return fingers  # [thumb, index, middle, ring, pinky]


prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    h, w, _ = img.shape
    now = time.time()
    action_text = ""
    finger_display = ""

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = fingers_up(handLms)
            total_up = fingers.count(1)
            thumb, index, middle, ring, pinky = fingers

            finger_display = f"T:{thumb} I:{index} M:{middle} R:{ring} P:{pinky}"

            # Reset gesture required: Open palm (all fingers up)
            if total_up == 5:
                gesture_reset_required = False
                current_gesture = None

            if not gesture_reset_required:
                # Priority order: Jump > Slide > Left > Right

                # JUMP: Only index up
                # JUMP: Index + Middle only
                if index == 1 and middle == 1 and thumb + ring + pinky == 0:

                    if now - last_action["jump"] > cooldown["jump"]:
                        pyautogui.press("up")
                        action_text = "Jump ⬆️"
                        last_action["jump"] = now
                        gesture_reset_required = True
                        current_gesture = "jump"
                        print("Gesture Detected: Jump")

                # SLIDE: Fist (all fingers down)
                elif total_up == 0:
                    if now - last_action["slide"] > cooldown["slide"]:
                        pyautogui.press("down")
                        action_text = "Slide 👇"
                        last_action["slide"] = now
                        gesture_reset_required = True
                        current_gesture = "slide"
                        print("Gesture Detected: Slide")

                # LEFT: Thumb + Index only
                elif thumb == 1 and index == 1 and middle + ring + pinky == 0:
                    if now - last_action["lr"] > cooldown["lr"]:
                        pyautogui.press("left")
                        action_text = "Left ⏪"
                        last_action["lr"] = now
                        gesture_reset_required = True
                        current_gesture = "left"
                        print("Gesture Detected: Left")

                # RIGHT: Thumb + Pinky only
                elif thumb == 1 and pinky == 1 and index + middle + ring == 0:
                    if now - last_action["lr"] > cooldown["lr"]:
                        pyautogui.press("right")
                        action_text = "Right ⏩"
                        last_action["lr"] = now
                        gesture_reset_required = True
                        current_gesture = "right"
                        print("Gesture Detected: Right")

                        # TOUCH: Thumb + Index + Middle only
                elif  middle == 1 and thumb+index+ring + pinky == 0:
                   if now - last_action["touch"] > cooldown["touch"]:
                    pyautogui.press("space")
                    action_text = "Touch (Start Game) ✨"
                    last_action["touch"] = now
                    gesture_reset_required = True
                    current_gesture = "touch"
                    print("Gesture Detected: Touch")


    # Overlay text
    cv2.putText(img, action_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(img, finger_display, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

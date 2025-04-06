import cv2
import mediapipe as mp

# open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# initailize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9) # detects only 1 hand & when 90% confident
mp_draw = mp.solutions.drawing_utils # for drawing hand landmarks

while True:
    ret, frame = cap.read() # read a frame
    if not ret:
        break

    frame = cv2.flip(frame, 1) # flip the frame horizontally (optional)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe expects RGB
    results = hands.process(rgb_frame) # process the RGB image to detect hands and get the landmarks

    # draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow("Hand Detection", frame) # show the frame in a window

    if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
        break

cap.release() # release the webcam
cv2.destroyAllWindows() # close the window
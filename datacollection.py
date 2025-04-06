import cv2
import mediapipe as mp

# open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# initailize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.9) # detects 2 hands & only when 90% confident
mp_draw = mp.solutions.drawing_utils # for drawing hand landmarks

while True:
    ret, frame = cap.read() # read a frame
    if not ret:
        break

    frame = cv2.flip(frame, 1) # flip the frame horizontally (optional)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe expects RGB
    results = hands.process(rgb_frame) # process the RGB image to detect hands and get the landmarks

    # draw hand landmarks
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # extract the label: 'Right' or 'Left'
            handedness = hand_info.classification[0].label
            
            # get bounding box around hand
            h, w, _ = frame.shape
            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            # draws rectangle around hand
            cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

            # displays the hand label
            cv2.putText(frame, handedness, (xmin, ymin - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
    cv2.imshow("Left and Right Hand Detection", frame) # show the frame in a window

    if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
        break

cap.release() # release the webcam
cv2.destroyAllWindows() # close the window
import cv2

# open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # read a frame
    if not ret:
        break

    frame = cv2.flip(frame, 1) # flip the frame horizontally (optional)
    cv2.imshow("webcam feed", frame) # show the frame in a window

    if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
        break

cap.release() # release the webcam
cv2.destroyAllWindows() # close the window
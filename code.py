import cv2
import numpy as np
def find_largest_contour(contours):
    max_area = -1
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour
def detect_hand_gesture(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("pexels-peter-fowler-6394054 (2160p).mp4")
        return
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = bg_subtractor.apply(frame)


        blurred = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_contour = find_largest_contour(contours)

        if hand_contour is not None:
            x, y, w, h = cv2.boundingRect(hand_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path ='pexels-peter-fowler-6394054 (2160p).mp4' # Specify the path to your video file
    detect_hand_gesture(video_path)

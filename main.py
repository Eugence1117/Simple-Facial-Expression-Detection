from deepface import DeepFace
import numpy as np
import cv2
import argparse
import mediapipe as mp

import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("-v","--video",help="Path to video file.")
    args = vars(arg.parse_args())

    if args.get("video",None) is None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(args.get("video"))
    pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    while video.isOpened():
        ret, frame = video.read()
        # _, frame2 = video.read()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            # Draw the pose annotation on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Find diff between frame to check for motion
            # diff = cv2.absdiff(frame, frame2)
            # _, threshold = cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), (5, 5), 0), 20, 255,
            #                              cv2.THRESH_BINARY)
            # dilated = cv2.dilate(threshold, None, iterations=3)
            # contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #
            # for contour in contours:
            #     (x, y, w, h) = cv2.boundingRect(contour)
            #     if cv2.contourArea(contour) < 900:
            #         continue
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Analytic Facial Expression
            analytics = DeepFace.analyze(frame, actions=["emotion"])
            region = analytics["region"]
            cv2.rectangle(frame, (region["x"], region["y"]), (region["x"] + region["w"], region["y"] + region["h"]),
                          (0, 0, 255), 1)

            cv2.putText(frame, analytics["dominant_emotion"], (region["x"], region["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)
        except ValueError:
            print("No Face detected")
        finally:
            cv2.imshow('frame', frame)
        # DeepFace.stream();
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Run')
    main()

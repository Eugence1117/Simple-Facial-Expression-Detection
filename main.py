from deepface import DeepFace
import numpy as np
import cv2

def main():
    video = cv2.VideoCapture(0)

    while(video.isOpened()):
        ret,frame = video.read()
        try:
            # Anala
            analytics = DeepFace.analyze(frame,actions=["emotion"])
            region = analytics["region"]
            cv2.rectangle(frame,(region["x"],region["y"]),(region["x"]+region["w"],region["y"]+region["h"]),(0,0,255),1)

            cv2.putText(frame, analytics["dominant_emotion"], (region["x"],region["y"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow('frame', frame)
        except ValueError:
            print("No Face detected")
        # DeepFace.stream();
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    print('Run')
    main()

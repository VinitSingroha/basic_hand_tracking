import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHAnds = mp.solutions.hands
        self.hands = self.mpHAnds.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findhand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHAnds.HAND_CONNECTIONS)
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(imgRGB)
                    for id, lm in enumerate(hand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 12:
                            cv2.circle(img, (cx, cy), 25, (255, 0, 255), 4)
                    # print(results.multi_hand_landmarks)
                    # if results.multi_hand_landmarks:
                    #     for hand in results.multi_hand_landmarks:
                    #         self.mpDraw.draw_landmarks(img, hand, self.mpHAnds.HAND_CONNECTIONS)

        return img



def main():
    cap = cv2.VideoCapture(0)
    flag = True

    detector = handDetector()
    while flag == True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findhand(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

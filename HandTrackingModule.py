import cv2
import mediapipe as mp
import time
import operator
from google.protobuf.json_format import MessageToDict

class HandDetector():
  def __init__(self, mode = False, maxHands = 2, complexity = 1, detectionConfidence = 0.5, trackingConfidence = 0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.complexity = complexity
    self.detectionConfidence = detectionConfidence
    self.trackingConfidence = trackingConfidence

    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionConfidence, self.trackingConfidence)
    self.mpDraw = mp.solutions.drawing_utils

  def findHands(self, img, draw =True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if self.results.multi_hand_landmarks:
      for self.handLms in self.results.multi_hand_landmarks:
        # print(self.results.multi_handedness)
        labelList = []
        for hand_handedness in self.results.multi_handedness:
          handedness_dict = MessageToDict(hand_handedness)
          labelList.append(handedness_dict["classification"][0]["label"])
        if draw:
          self.mpDraw.draw_landmarks(img, self.handLms, self.mpHands.HAND_CONNECTIONS)
      print(labelList)
    return img

  def findPosition(self, img, handNo = 0, draw=True):
    lmList = []
    if self.results.multi_hand_landmarks:
      myHand = self.results.multi_hand_landmarks[handNo]

      for id, lm in enumerate(self.handLms.landmark):
        # print(id,lm)
        h, w, c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        # print(id, cx, cy)
        lmList.append([id, cx, cy])
        if draw:
          cv2.circle(img, (cx, cy), 7, (255, 255, 255), cv2.FILLED)
    return lmList




def main():
  pTime = 0
  cTime = 0
  cap = cv2.VideoCapture(0)
  detector = HandDetector()

  while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw = True)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
      # print(lmList)
      print(max(lmList, key = operator.itemgetter(1)))
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
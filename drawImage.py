import cv2
import main as htm

detector = htm.HandDetector()

files = ["fake.png"]
for file in files:
  img = cv2.imread(f"img/{file}")
  img = detector.findHands(img)
  # lmList = detector.findPosition(img)
  # if len(lmList) != 0:
  #   print(lmList)
  cv2.imshow("Image", img)
  k=cv2.waitKey(0) & 0xFF
  if k== ord("s"):  # wait for 'q' key to save and display next image
    cv2.imwrite(f"img/draw/{file}",img)
    cv2.destroyAllWindows()
  else:
    cv2.destroyAllWindows()
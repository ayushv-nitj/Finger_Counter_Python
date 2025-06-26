import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    print (f'{folderPath}/{imPath}')

    overlayList.append(image)
print(len(overlayList))

pTime = 0
totalFingers = 0

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if(len(lmList) != 0):
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)


    #Four fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        #print(totalFingers)

        # Resize the overlay image to fit into 200x200 region
        overlaySmall = cv2.resize(overlayList[totalFingers - 1], (150, 200))  # (width, height)

        # Paste resized image onto the webcam frame
        img[0:200, 0:150] = overlaySmall

        h, w, c = overlayList[totalFingers - 1].shape
        if h <= img.shape[0] and w <= img.shape[1]:
            img[0:h, 0:w] = overlayList[totalFingers - 1]


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
    cv2.putText(img, f'FPS: {int(fps)}', (30,260), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.putText(img, f' {(totalFingers)}', (-75,420), cv2.FONT_HERSHEY_PLAIN, 13, (255,0,0), 20)

    #cv2.putText(img, f'Count: {(totalFingers)}', (10,300), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
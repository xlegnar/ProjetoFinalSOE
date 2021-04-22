import cv2
import numpy as np
from PIL import Image

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_alt2.xml"
faceDetect = cv2.CascadeClassifier(cascadePath);
id=0

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
while True:
    ret, img =cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = rec.predict(gray[y:y+h,x:x+w])
        if(conf >= 60):
            if(Id==1):
                Id="Vitor"
            elif(Id==2):
                Id="Thau"
        else:
            Id="Unknowm"
        cv2.putText(img,str(Id), (x,y+h), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('im',img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
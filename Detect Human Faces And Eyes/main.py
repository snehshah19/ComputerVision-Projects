import cv2

# load the classifier with the path and the xml files

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors = 5, minSize = (30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors = 10, minSize = (15,15))

    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)   

    cv2.imshow("real time face and eyes detection",frame)         

    if(cv2.waitKey(1) & 0xFF == ord('q') ):
        break

cap.release()
cv2.destroyAllWindows()    


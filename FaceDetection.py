import cv2
face_cascade =cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame,scaleFactor=1.05,minNeighbors=5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #img = cv2.circle(frame,(int(x+w/2),int(y+h/2)),int(w/1.4),(255,0,255),thickness=3)
        roi_frame = frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(frame[y:y+h, x:x+w])
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.release()
cv2.destroyAllWindows()
import cv2 #pip install opencv-python

harCascade = r"C:\Users\maisu\Desktop\Personal AI Projects\Face Detection\model\haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)

cap.set(3,640) #width
cap.set(4,480) #height

while True:
    suceed , image = cap.read()

    faceCascade = cv2.CascadeClassifier(harCascade)
    image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(image_gray , 1.1 , 4)

    for (x,y,w,h) in face:
        cv2.rectangle(image , (x,y) , (x+w , y+h) , (0,255,0) , 2)

    # cv2.imshow("face" , image) --> for rgb color camera
    cv2.imshow("face" , image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
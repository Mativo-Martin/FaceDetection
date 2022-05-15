import cv2 #This is a module which allows one to carry out computer vision with python 

#we then store the xml file in a variable
#the file  contains all the info required by a computer in order to detect a face upon receiving an image input
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


img = cv2.imread('morgan.jpg')

#here we convert the image to greyscale 
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#we then get the coordinates of the rectangle to highlight the face
faces = face_cascade.detectMultiScale(gray_img,
    scaleFactor = 1.05, 
    minNeighbors = 5)

#draw a rectangle around the face using the coordinates 
#x,y are the coordinates for the top left vertice of the rectangle
#(x+w),(y+l) are the coordinates for the bottom right vertice 
for x, y, w, l in faces: 
    
    img = cv2.rectangle(img, (x,y), (x+w,y+l), (0,255,0), 2)

#display image 
cv2.imshow('Color',img)

#end the session
cv2.waitKey(0) 
cv2.destroyAllWindows()

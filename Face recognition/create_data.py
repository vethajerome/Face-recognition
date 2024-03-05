import cv2, os #os->for directory oriented functions,as face is captured and saved in a directory
haar_file = 'haarcascade_frontalface_default.xml' #loads haar cascade file
datasets = 'datasets'  #datasets folder holds the data 
sub_data = 'Jerome2'  #the folder inside the datasets are called as sub data,in this line we are to create a folder named "Elon"   

path = os.path.join(datasets, sub_data) #datasets/Ramesh
if not os.path.isdir(path): #if the folder does not exist create one 
    os.mkdir(path)#to create a new directory (folder)
(width, height) = (130, 100) #the required size to load   


face_cascade = cv2.CascadeClassifier(haar_file)#classifier to load the file

webcam = cv2.VideoCapture(0)  #camera initialisation

count = 1 
while count < 51: #higher the value , higher the accuracy
    print(count)
    (_, im) = webcam.read()#second value(im) captures frame
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)#to obtain the required face points
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)#to construct rectangle with the obtained coordinates
        face = gray[y:y + h, x:x + w]#to obtain the face pic with the  box
        face_resize = cv2.resize(face, (width, height))#to equal the pics
        cv2.imwrite('%s/%s.png' % (path,count), face_resize)#marks each index of the image as path
    count += 1#for incrementing(loop purpose)
	
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27: #27 escape key value 
        break
webcam.release()#releases the camera
cv2.destroyAllWindows()

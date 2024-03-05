import cv2, numpy, os #numpy is used for array conversion,as the dataset is present in a directory we use os
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)#images->the images in the dataset,labels->the indexes of the folder as per arrangement ,names->{Elon : 0},{Ramesh : 1}[keys:values]
for (subdirs, dirs, files) in os.walk(datasets):#folder->directory,inside it  are sub directory, then the pics are files
    for subdir in dirs:#taking the folders in loop 
        names[id] = subdir #the first subd is initialized as the first element 
        subjectpath = os.path.join(datasets, subdir)#binding the dataset and the subd into a single path 
        for filename in os.listdir(subjectpath):#to take each image from the subd
            path = subjectpath + '/' + filename #each image in each time of the loop gets the path
            label = id
            images.append(cv2.imread(path, 0)) #adds each image into the list by reading it 
            labels.append(int(label)) #to index the item added to the list 
            #print(labels)  so only after all the images in the folder is processed then the first enclosed loop moves to the next folder 
        id += 1  #for the next element to be in the array
(width, height) = (130, 100)
#so far initialisation is done 
(images, labels) = [numpy.array(lis) for lis in [images, labels]] #Array conversionn

#print(images, labels)
##model = cv2.face.LBPHFaceRecognizer_create()
model =  cv2.face.FisherFaceRecognizer_create() #the classifier for fisher face
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)#already the face is seperated out ,but now we are loading it again for the live face detection 
webcam = cv2.VideoCapture(0)
cnt=0 #to indicate an unknown face
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #converting gray scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)#to predict the output of the live face capture 
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1]<800: #prediction[1] indicates confidence level ,[0] determines its ID
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_COMPLEX,1,(51, 255, 255))#to display the name of the person and accuracy level 
            print (names[prediction[0]])#prints the name of the corresponding person 
            cnt=0
        else: #for a new face detected
            cnt+=1
            cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))#prints as unknown 
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("input.jpg",im)#saves the unknown face
                cnt=0
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()

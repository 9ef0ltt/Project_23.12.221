import cv2


cap = cv2.VideoCapture('woman.mp4') #ввод видео

while True:
    success, img = cap.read()

    gh = cv2.CascadeClassifier("eyes.xml") #используем нейронную сеть, для определения глаз
    results = gh.detectMultiScale(img, scaleFactor=2, minNeighbors=8)# привязка к глазам (scaleFactor-меньшее число, minNeighbors-любое число

    print(results) #вывод
    for (x, y, w, h) in results:
        # обводка глаз и размытие глаз
        cv2.rectangle(img, (x,y),(x+w, y+h),(0,0,0),thickness=2) #обводим глаз в квадрат
        sub_face = img[y:y + h, x:x + w]
        sub_face  = cv2.GaussianBlur(sub_face, (101, 101), 0) #размытие глаз
        img[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face
    cv2.imshow('Resut', img) #вывод конечного видео

    if cv2.waitKey(1) & 0xFF == ord('q'): #выход из видео - нажмите Q
        break
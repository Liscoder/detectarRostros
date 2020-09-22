import cv2
import os #importa y crea direcctorios 

imagesPath = "C:\git\detectarrostros\images" # Cambia a la ruta donde hayas almacenado la carpeta con las imágenes
imagesPathList = os.listdir(imagesPath)

if not os.path.exists('Rostros encontrados'): #no existe una carpeta con ese nombre
	print('Carpeta creada: Rostros encontrados')
	os.makedirs('Rostros encontrados')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0
for imageName in imagesPathList:
    #print('imageName', imageName)
	image = cv2.imread(imagesPath+'/'+imageName)
	imageAux = image.copy() #copia de la imagen
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	faces = faceClassif.detectMultiScale(gray, 1.1, 5)  #almacena en faces

	for (x,y,w,h) in faces:   
		cv2.rectangle(image, (x,y),(x+w,y+h),(128,0,255),2)
	cv2.rectangle(image,(10,5),(450,25),(255,255,255),-1)
	cv2.putText(image,'Presione s, para almacenar los rostros encontrados',(10,20), 2, 0.5,(255,211,0),1,cv2.LINE_AA)
	cv2.imshow('image',image)
	k = cv2.waitKey(0)
	if  k == ord('s'):
		for (x,y,w,h) in faces:
			rostro = imageAux[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
			#cv2.imshow('rostro',rostro)
			#cv2.waitKey(0)
			cv2.imwrite('Rostros encontrados/rostro_{}.jpg'.format(count),rostro)
			count = count +1
	elif k == 27:
		break

cv2.destroyAllWindows()
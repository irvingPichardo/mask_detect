# USAGE
# python3 detect_mask_video.py

# importar los paquetes necesarios
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
	# agarra las dimensiones del marco y luego construye un blob
	# de eso
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pasar el blob a través de la red y obtener las detecciones faciales
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# inicializar nuestra lista de caras, sus ubicaciones correspondientes,
	# y la lista de predicciones de nuestra red de mascarillas

	faces = []
	locs = []
	preds = []

	# bucle sobre las detecciones
	for i in range(0, detections.shape[2]):
		# extraer la confianza (es decir, la probabilidad) asociada con
		# la detección
		confidence = detections[0, 0, i, 2]

		# filtrar las detecciones débiles asegurando que la confianza es
		# mayor que la confianza mínima
		if confidence > args["confidence"]:
			# calcular las coordenadas (x, y) del cuadro delimitador para
			# el objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# asegúrese de que los cuadros delimitadores caigan dentro de las dimensiones de
			# el marco
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extraer el ROI de la cara, convertirlo de BGR a canal RGB
			# pedidos, redimensionarlo a 224x224 y preprocesarlo
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# agregue la cara y los cuadros delimitadores a sus respectivos
			# listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# solo hace predicciones si se detectó al menos una cara
	if len(faces) > 0:
		# para una inferencia más rápida, haremos predicciones por lotes en * todos *
		# caras al mismo tiempo en lugar de predicciones una por una
		# en el bucle `for` anterior
		preds = maskNet.predict(faces)

	# devolver una tupla de 2 de las ubicaciones de la cara y sus correspondientes
	# ubicaciones
	return (locs, preds)

# construir el analizador de argumentos y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# cargar nuestro modelo de detector de cara serializado desde el disco
print("[INFO] cargando el modelo de detector de cara... \U0001F637  \U0001F637")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# cargar el modelo de detector de máscara facial desde el disco
print("[INFO] cargando el modelo de detector de máscara facial... \U0001F637  ")
maskNet = load_model(args["model"])

# inicializa la transmisión de video y permite que el sensor de la cámara se caliente
print("[INFO]  iniciando transmisión de video...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# bucle sobre los cuadros de la secuencia de video
while True:
	# agarra el cuadro de la secuencia de video enhebrada y redimensiona
	# para tener un ancho máximo de 400 píxeles que traía por defecto el programa
	#pero la ventana se veía muy pequeña en mi computadora, así que lo aumenté a 910pix 
	frame = vs.read()
	frame = imutils.resize(frame, width=910)

	# detecta rostros en el marco y determina si están usando un
	# mascarilla o no
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	
	# bucle sobre las ubicaciones de caras detectadas y sus correspondientes
	# ubicaciones
	for (box, pred) in zip(locs, preds):
		# desempaquetar el cuadro delimitador y las predicciones
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinar la etiqueta de clase y el color que usaremos para dibujar
		# el cuadro delimitador y el texto
		label = "Tienes cubreboca" if mask > withoutMask else "No Tienes cubreboca"
		color = (0, 255, 0) if label == "Tienes cubreboca" else (0, 0, 255)

		# incluye la probabilidad en la etiqueta
		label = "{}: Con {:.2f}%".format(label, max(mask, withoutMask) * 100)+" de probabilidad"


		import serial 
		arduino=serial.Serial(port='/dev/cu.usbmodem14201', baudrate=9600)
		if mask==1:
			arduino.write(mask)
		else:
			withoutMask==0
			arduino.write(withoutMask)

		# muestra la etiqueta y el rectángulo del cuadro delimitador en la salida
		# marco
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# muestra el marco de salida
	cv2.imshow("Detector de Cubrebocas *2020*",frame)
	key = cv2.waitKey(1) & 0xFF

	# si se presionó la tecla `q`, salga del bucle
	if key == ord("q"):
		break

	# si se presionó la tecla `q`, salga del bucle
cv2.destroyAllWindows()
vs.stop()
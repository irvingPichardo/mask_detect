# python3 detect_mask_image.py --image examples/example_01.png

# importar los paquetes necesarios

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# construir el analizador de argumentos y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
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
print("[INFO]  cargando el modelo de detector de cara... \U0001F637  \U0001F637 ")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# cargar el modelo de detector de máscara facial o cubreboca desde el disco
print("[INFO] cargando el modelo de detector de máscara facial... \U0001F637")
model = load_model(args["model"])

# cargar la imagen de entrada desde el disco, clonarla y tomar la imagen espacial
# dimensiones
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construir un blob de la imagen
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# pasar el blob a través de la red y obtener las detecciones faciales
print("[INFO] calculando detecciones faciales... \U0001F637")
net.setInput(blob)
detections = net.forward()

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
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pasar la cara a través del modelo para determinar si la cara
		# tiene una máscara o no
		(mask, withoutMask) = model.predict(face)[0]

		# determinar la etiqueta de clase y el color que usaremos para dibujar
		# el cuadro delimitador y el texto
		label = "Tienes Cubrebocas" if mask > withoutMask else "No Tienes Cubrebocas"
		color = (0, 255, 0) if label == "Tienes Cubrebocas" else (0, 0, 255)

		# incluye la probabilidad en la etiqueta
		label = "{}: Con {:.2f}%".format(label, max(mask, withoutMask) * 100)+" de probabilidad"

		# muestra la etiqueta y el rectángulo del cuadro delimitador en la salida
		# marco
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# muestra la imagen de salida
cv2.imshow("Resultado", image)
cv2.waitKey(0)
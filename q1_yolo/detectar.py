from ultralytics import YOLO
import cv2

modelo = YOLO("yolov8n.pt") #versoes: v8n e v8s

CLASSE_ALVO = "car"
CONFI_MIN = 0.4

resultado = modelo(
    "imagens/estacionamentoc.jpg", conf = CONFI_MIN, iou = 0.5      
)[0]

confs = []
contador = 0
for box in resultado.boxes:
    classe_nome = modelo.names[int(box.cls)]
    if classe_nome == CLASSE_ALVO:
        contador += 1
        confs.append(float(box.conf))

print(f"Total de '{CLASSE_ALVO}' detectados: {contador}")
if confs:
    print(f"Confiança média: {sum(confs)/len(confs):.3f}")

# Salva imagem anotada (todas as classes, ou filtre se quiser só CLASSE_ALVO)
imagem_anotada = resultado.plot()
cv2.imwrite("saida/estacionamento_anotado_v8n.jpg", imagem_anotada)
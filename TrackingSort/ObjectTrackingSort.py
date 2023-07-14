# %%
import sys
import os
import centernet

# sys.path.append("TrackingSort")
from sort import Sort
from lib import VisTrack, show_video, create_video

import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import shutil
# %%
video_file = "data-input/video.mp4"
show_video(video_file)

# %%
# Default: num_classes=80
obj = centernet.ObjectDetection(num_classes=80)
# num_classes=80 and weights_path=None: Pre-trained COCO model will be loaded.
obj.load_weights(weights_path=None)

# %%
# EXAMPLE IMAGE
p_img = PIL.Image.open('data-input/img.jpeg')
img = cv2.cvtColor(np.array(p_img), cv2.COLOR_BGR2RGB)

boxes, classes, scores = obj.predict(img)

# Initialize the vision tracker
vt = VisTrack()
vt.draw_bounding_boxes(p_img, boxes, classes, scores)

# %%
##### PREDICT ON VIDEO ##########
vidcap = cv2.VideoCapture(video_file)
fps = vidcap.get(cv2.CAP_PROP_FPS)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

folder_out = "Track"
if os.path.exists(folder_out):
    shutil.rmtree(folder_out)
os.makedirs(folder_out)

draw_imgs = []

# Tracking algorithm
sort = Sort(max_age=1, min_hits= 3, iou_threshold = 0.3)

pbar = tqdm(total = length)
i = 0
while True:
    ret, frame = vidcap.read()
    if not ret:
        break

    boxes, classes, scores = obj.predict(frame)
    detections_in_frame = len(boxes)
    if detections_in_frame:
        # centernet will do detection in all COCO classes. "person" is class 0
        idxs = np.where(classes == 0)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]
        classes = classes[idxs]
    else:
        boxes = np.empty((0,5)) 

    dets = np.hstack((boxes, scores[:, np.newaxis]))

    #Update Sort algorithm to track bounding boxes using Kalman Filter
    res = sort.update(dets)

    boxes_track = res[:,:-1]
    boxes_id = res[:, -1].astype(int)

    p_frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if detections_in_frame:
        p_frame = vt.draw_bounding_boxes(p_frame, boxes_track, boxes_id, scores)
    p_frame.save(os.path.join(folder_out, f"{i:03d}.png"))

    i+=1
    pbar.update(1)
# %%
############# CREATE THE VIDEO ####################
track_video_file = 'tracking'

def crear_video(folder_out, nombre_video, fps=25):
    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(folder_out)
    archivos.sort()  # Ordenar los archivos en orden alfabético

    # Leer la primera imagen para obtener el tamaño del video
    ruta_primera_imagen = os.path.join(folder_out, archivos[0])
    imagen = cv2.imread(ruta_primera_imagen)
    alto, ancho, _ = imagen.shape

    # Crear el objeto VideoWriter
    ruta_video = nombre_video + ".avi"
    video = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (ancho, alto))

    # Recorrer todas las imágenes y agregarlas al video
    for archivo in archivos:
        ruta_imagen = os.path.join(folder_out, archivo)
        imagen = cv2.imread(ruta_imagen)
        video.write(imagen)

    # Liberar el objeto VideoWriter y mostrar un mensaje de finalización
    video.release()
    print("Se ha creado el video:", ruta_video)

crear_video(folder_out, track_video_file, fps = fps)
# %%
import sys
import os
import centernet

sys.path.append("TrackingSort\ObjectTrackingSort.py")
from sort import Sort
from lib import VisTrack, show_video, create_video

import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import shutil
# %%
video_file = "TrackingSort\video.mp4"

# %%
vt = VisTrack()

# Default: num_classes=80
obj = centernet.ObjectDetection(num_classes=80)

# num_classes=80 and weights_path=None: Pre-trained COCO model will be loaded.
obj.load_weights(weights_path=None)
# %%
p_img = PIL.Image.open('TrackingSort\img.jpeg')
img = cv2.cvtColor(np.array(p_img), cv2.COLOR_BGR2RGB)

boxes, classes, scores = obj.predict(img)

vt.draw_bounding_boxes(p_img, boxes, classes, scores)
# %%

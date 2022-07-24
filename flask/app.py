import os
os.chdir("/home/asyed/my_docker/Yolov5_DeepSort_Pytorch")
import track
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import imageio
from track import detect 
from flask import Flask, render_template, Response
import time
from subprocess import call
import threading
import time
import  numpy as np
from flask_bootstrap import Bootstrap
from functools import lru_cache
import json
from time import time
from random import random
import numpy as numpy
import json
from flask import Flask, render_template, make_response
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt



agnostic_nms = False
augment = False
classes = None
conf_thres = 0.3
config_deepsort = '/home/asyed/my_docker/Yolov5_DeepSort_Pytorch/deep_sort/configs/deep_sort.yaml'
deep_sort_model = 'osnet_x0_25'
device = 'cuda'
dnn = False
evaluate = False
exist_ok = False
fourcc = 'mp4v'
half = False
imgsz = [640, 480]
iou_thres = 0.5;
max_det = 1000;
name = 'exp'
# output='inference/output'; project=PosixPath('runs/track');save_txt=False,
save_vid = True;
show_vid = False,
source = 'rtsp://10.41.29.20/?video=2'
# source = 'https://www.youtube.com/watch?v=AdUw5RdyZxI'
# source1 = 'rtsp://10.41.29.19/?video=2'
visualize = False;
yolo_model = "/home/asyed/my_docker/Yolov5_DeepSort_Pytorch/yolov5m.pt"
webcam = True
save_vid = True
save_txt = True

h = np.array([[-4.32821609e-01, -9.57458673e+00,  5.28269104e+03],
        [ 9.00846886e+00,  3.04434022e+01, -4.21910654e+03],
        [ 3.41446237e-04,  8.82611821e-02,  1.00000000e+00]])

# h = np.array([[-1.05784263e-01, -4.93486191e-01,  4.17952058e+02],
#        [-7.90342192e-02, -5.35174570e-01,  3.81509467e+02],
#        [-1.45909328e-04, -1.47646344e-03,  1.00000000e+00]])
def convert_bev(arr):
    #p = np.array(arr).reshape(3,1)
    p = arr.reshape(3,1)
    temp_p = h.dot(p)
    sum = np.sum(temp_p ,1)
    px = int(round(sum[0]/sum[2]))
    py = int(round(sum[1]/sum[2]))
#     print(px)
    return np.array([px,py])

app = Flask(__name__)

Bootstrap(app)

cfg = get_config()
cfg.merge_from_file(config_deepsort)

deepsort = DeepSort(deep_sort_model,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

device = select_device(device)
half &= device.type != 'cpu'  # half precision only supported on CUDA

# The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
# its own .txt file. Hence, in that case, the output folder is not restored
# make new output folder


# Load model
device = select_device(device)
model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt:
    model.model.half() if half else model.model.float()

# Set Dataloader
vid_path, vid_writer = None, None
# Check if environment supports image displays

cudnn.benchmark = True  # set True to speed up constant image size inference

dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)

# dataset1 = LoadStreams(source1, img_size=imgsz, stride=stride, auto=pt and not jit)



bs = len(dataset)  # batch_size

vid_path, vid_writer = [None] * bs, [None] * bs

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names

if pt and device.type != 'cpu':
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def my_gen():
    # global framess_im2

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # arr = None
    
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
       
        
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        # print("raw_frame",img.shape)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        pred = model(img, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

    # Process detections

        # dets_per_img = []
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count

                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
    # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

# Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

# pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                dets_per_img = []

                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # bbox_left = bbox_left + bbox_h
                            bbox_top = bbox_top + bbox_h

                            
                            pts = np.array([[bbox_left, bbox_top, 1]])
                            arr_per = convert_bev(pts)
                            # print("arr_per_to", arr_per)


                            arr_per = np.append(id, arr_per)
                            # print("arr_per1_to", arr_per1)

                            dets_per_img.append(arr_per)
                         
                         
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                arr_per= None
                dets_per_img =[None, None]
                LOGGER.info('No detections')
            
            im0 = annotator.result()

            if len(dets_per_img) > 1:
                arr_per = np.stack(dets_per_img).tolist()
            
            elif len(dets_per_img) == 1:
                arr_per = np.array([dets_per_img[0]])
                # print("not_stack",arr_per)

            elif dets_per_img is None:
                arr_per = None
            
            if save_vid:
        
                fps, w, h = 30, im0.shape[1], im0.shape[0]
                cv2.putText(im0, str(frame_idx), (500,460) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                framess = cv2.imencode('.jpg', im0)[1].tobytes()  # Remove this line for test camera
                if arr_per is None:
                    yield framess
                else:
                    yield framess, arr_per
                
                        
def bev_map():
    global bev_arr
    for _, fr in enumerate(my_gen()):
        if len(fr) < 2:
            bev_arr = None
            yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + fr + b'\r\n')

        else:

            det , bev_arr = fr
            yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + det + b'\r\n')
                     
    
def bev_map1():
     cnt = 0

     while True:

        cnt = cnt + 1



        bev_img = cv2.imread("/home/asyed/my_docker/Yolov5_DeepSort_Pytorch/map_harmons.png")

        
        for coords in bev_arr:
            print("coords_with_id", coords)

            if coords is None:
                break

            coords_only = coords[1:]

            print("coords", coords)
            # print("coords_shape", coords.shape)

            cv2.putText(bev_img, str(cnt), (500,400) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
         #   cv2.circle(bev_img, tuple(bev_arr[0]), 2,(255,0,0), 5)
            
            cv2.putText(bev_img, str(coords[0]), tuple(coords_only) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.circle(bev_img, tuple(coords_only), 2,(255,0,0), 5)

        be_enc = cv2.imencode('.jpg', bev_img)[1].tobytes() 

        yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + be_enc+ b'\r\n')


    

@app.route('/video_feed')

def video_feed():
    return Response(bev_map(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(bev_map1(), mimetype='multipart/x-mixed-replace; boundary=frame')


def checker_bev():
   while True:
       #bev_map()
       video_feed1()
       # data()
       time.sleep(0.002)


if __name__ == '__main__':
    
    x = threading.Thread(target=checker_bev, daemon = True)
    x.start()
    
    app.run(host='0.0.0.0', threaded=True, debug = True)













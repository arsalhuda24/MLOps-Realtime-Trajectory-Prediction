import os
# os.chdir("/home/asyed/my_docker/Yolov5_DeepSort_Pytorch")
#import track
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
# sys.path.insert(0, './yolov5')
sys.path.insert(0, '/home/asyed/airflow/dags/yolov5')
sys.path.insert(0, '/home/asyed/airflow/dags')

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
#from track import detect 
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
import io
from pymongo import MongoClient
from datetime import date

import warnings
warnings.filterwarnings("ignore")


def rtsp_to_mongodb():

    with open("/home/asyed/airflow/dags/parameters.json") as f:

        parms = json.load(f)

    agnostic_nms = parms["agnostic_nms"]
    augment =  parms["augment"]
    classes = parms["classes"]
    conf_thres = parms["conf_thres"]
    config_deepsort = parms["config_deepsort"]
    deep_sort_model = parms["deep_sort_model"]
    device = parms["device"]
    dnn = False
    evaluate = parms["evaluate"]
    exist_ok = parms["exist_ok"]
    fourcc = parms["fourcc"] 
    half = False
    print(device)
    imgsz = parms["imgsz"]
    iou_thres = parms["iou_thres"]
    max_det = parms["max_det"]
    name = parms["name"]
   # save_vid = parms["save_vid"]
    #show_vid = parms["show_vid"]
    source = parms["source"]
    visualize = parms["visualize"]
    yolo_model = parms["yolo_model"]
    webcam = parms["webcam"]
    save_txt = parms["save_txt"]
    homography = np.array(parms["homography"])


    url = "mongodb://localhost:27017" 
    client = MongoClient(url)
    db = client.trajectory_database

    today_date = date.today().strftime("%m-%d-%y")
    new = "file_image_coordinates_"+today_date
    collection = db[new]

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


    bs = len(dataset)  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        # global framess_im2

        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        # arr = None
        past = []
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

                        
                                agent_data = { 'frame' : int(frame_idx +1),
                                                 'agent_id':int(id),
                                                 "labels": str(names[c]),
                                                 "x" : int(bbox_left),
                                                 "y" : int(bbox_top)

                                 }

                                print("agent",agent_data)

                                collection.insert_one(agent_data)

                                #db.object_detection.insert_one(agent_data)
                                #db.pedestrian_detection_15_june.insert_one(agent_data)
                                #db.test_21_july.insert_one(agent_data)

                                                   
                             
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

                else:
                    deepsort.increment_ages()               
                    LOGGER.info('No detections')
                
                im0 = annotator.result()
                    

if __name__ == '__main__':

    rtsp_to_mongodb()












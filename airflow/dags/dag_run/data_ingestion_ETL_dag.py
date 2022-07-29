from datetime import datetime, timedelta
from textwrap import dedent
import os 
import sys
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# sys.path.insert(0,'/home/asyed/airflow/dags' )
# sys.path.insert(0, '/home/asyed/airflow/dags/yolov5')

# from data_ingest import rtsp_to_mongodb
# import torch
# import torch.backends.cudnn as cudnn

# from yolov5.models.experimental import attempt_load
# from yolov5.utils.downloads import attempt_download
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.datasets import LoadImages, LoadStreams
# from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
#                                   check_imshow, xyxy2xywh, increment_path)
# from yolov5.utils.torch_utils import select_device, time_sync
# from yolov5.utils.plots import Annotator, colors
# from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort
# import imageio
# #from track import detect 
# from flask import Flask, render_template, Response
# import time
# from subprocess import call
# import threading
# import time
# import  numpy as np
# from flask_bootstrap import Bootstrap
# from functools import lru_cache
# import json
# from time import time
# from random import random
# import io
# from pymongo import MongoClient
# from datetime import date
# import json 



# with open("parameters.json") as f:

#     parms = json.load(f)

#     agnostic_nms = parms["agnostic_nms"]
#     augment =  parms["augment"]
#     classes = parms["classes"]
#     conf_thres = parms["conf_thres"]
#     config_deepsort = parms["config_deepsort"]
#     deep_sort_model = parms["deep_sort_model"]
#     device = parms["device"]
#     dnn = parms["dnn"]
#     evaluate = parms["evaluate"]
#     exist_ok = parms["exist_ok"]
#     fourcc = parms["fourcc"] 
#     half = parms["half"]
#     imgsz = parms["imgz"]
#     iou_thres = parms["iou_thres"]
#     max_det = parms["max_det"]
#     name = parms["name"]
#     save_vid = parms["save_vid"]
#     show_vid = parms["show_vid"]
#     source = parms["source"]
#     visualize = parms["visualize"]
#     yolo_model = parms["yolo_model"]
#     webcam = parms["webcam"]
#     save_txt = parms["save_txt"]
#     homography = np.array(parameters["homography"])

# rtsp_to_mongodb()



with DAG(
    'tutorial',
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        'depends_on_past': False,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function,
        # 'on_success_callback': some_other_function,
        # 'on_retry_callback': another_function,
        # 'sla_miss_callback': yet_another_function,
        # 'trigger_rule': 'all_success'
    },
    description='detection_tracking',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2022, 7, 28),
    catchup=False,
    tags=['example'],
) as dag:


    #t_get_traj = PythonOperator(task_id = "get_bb", python_callable = rtsp_to_mongodb)

    t2 = BashOperator( task_id="detect_track",
    # "test.sh" is a file under "/opt/scripts"
                    bash_command= "/home/asyed/anaconda3/envs/yolo_deepsort/bin/python /home/asyed/airflow/dags/data_ingest.py",
                    dag=dag,
)

    # t1, t2 and t3 are examples of tasks created by instantiating operators
    # t1 = BashOperator(
    #     task_id='print_date',
    #     bash_command='date',
    # )

    # t2 = BashOperator(
    #     task_id='sleep',
    #     depends_on_past=False,
    #     bash_command='sleep 5',
    #     retries=3,
    # )
    # t1.doc_md = dedent(
    #     """\
    # #### Task Documentation
    # You can document your task using the attributes `doc_md` (markdown),
    # `doc` (plain text), `doc_rst`, `doc_json`, `doc_yaml` which gets
    # rendered in the UI's Task Instance Details page.
    # ![img](http://montcs.bloomu.edu/~bobmon/Semesters/2012-01/491/import%20soul.png)

    # """
    # )

    # dag.doc_md = __doc__  # providing that you have a docstring at the beginning of the DAG
    # dag.doc_md = """
    # This is a documentation placed anywhere
    # """  # otherwise, type it like this
    # templated_command = dedent(
    #     """
    # {% for i in range(5) %}
    #     echo "{{ ds }}"
    #     echo "{{ macros.ds_add(ds, 7)}}"
    # {% endfor %}
    # """
    # )

    # t3 = BashOperator(
    #     task_id='templated',
    #     depends_on_past=False,
    #     bash_command=templated_command,
    # )

    # t1 >> [t2, t3]
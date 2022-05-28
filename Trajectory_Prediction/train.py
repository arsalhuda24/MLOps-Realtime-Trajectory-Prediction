from preprocess import TrajectoryDataset
import numpy as np
import pandas as pd
import tensorflow as tf



path = "/home/asyed/my_docker/Yolov5_DeepSort_Pytorch/runs/track/exp91/train"
dset = TrajectoryDataset(
        path,
        obs_len=8,
        pred_len=12,
        skip=1,
        delim=' ')


train_x = dset.obs_traj[:,2:,:].permute(0,2,1)
train_x[:,:,0] = train_x[:,:,0] /640
train_x[:,:,1] = train_x[:,:,1] /480

train_y = dset.pred_traj[:,2:,:].permute(0,2,1)
train_yn = train_y
train_yn[:,:,0] = train_y[:,:,0] /640
train_yn[:,:,1] = train_y[:,:,1] /480



train_xx = tf.convert_to_tensor(
    train_x, dtype=None, dtype_hint=None, name=None
)

train_yy = tf.convert_to_tensor(
    train_y, dtype=None, dtype_hint=None, name=None
)


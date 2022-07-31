# Real-Time Trajectory Prediction System
<!-- <p align="center">
  <img width="800" height="250" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/detection_yolo5_deepsort.gif">
</p> -->

<p align="center">
  <img width="560" height="250" src="https://github.com/arsalhuda24/MLOps-Realtime-Trajectory-Prediction/blob/master/images/bev_trajs.gif">
</p>

## Overview
<p align="center">
  <img width="1000" height="250" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/system_design.jpg">
</p>
This project aims to forecast future trajectories of pedestrian in a scene. The trajectory prediction task works as follows: given the past observed trajectory of an agent the goal is to predict the future trajectory (coordinates) of the agent in subsequent frames. Following the literature norm, the prediction task is commonly tackled for 8 observed positions and 12 predicted positions. The time between two positions being 0.4s, we observe an agent moving during 3.2s and predict its movement for the next 4.8s. In order to generate robust realtime predictions we build a continious learning infrastructure which will have following components. 

- ETL (Extract , Transform, Load) Pipline  
- Model Development (LSTM, GAN, VAE)
- Deployment 
- Realtime Inference
- Monitoring 
- Re-training  

## Data Ingestion and ETL Pipeline 
<p align="center">
  <img width="1000" height="220" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/ingestion_etl_pipeline.jpg">
</p>

- Please follow this [link](https://www.digitalocean.com/community/tutorials/how-to-install-mongodb-on-ubuntu-20-04) to set up MongoDB and install MongoDB Compass client. 

- Follow this [link](https://www.hackerxone.com/2021/10/15/steps-to-install-and-setup-apache-airflow-on-ubuntu-20-04-lts/) to set up Apache Airflow. This will help to automate ETL tasks and load new stream of data into MongoDB

We capture RTSP video streams throuugh openCV and run Yolov5 and Deepsort for detection and tracking of pedestrians in the scene. The image coordintaes are then transofrmed in to world coorinates by using homography matrix (I will post a sepetae tutorial on computing homography). After that, trajectory sequences are generated and stored in mongodb. This serves as a feature stroe for training prediciton models



    .
    ├── airflow                   
      ├── ETL                    # source code for data extraction, transformation and storing.
      ├── dags                   # Dags for running ETL jobs 
      ├── logs
      ├── airflow-webserver.pid
      ├── airflow.cfg
    ├── flask                     # code for inference
    ├── images                    
    ├── trajectory_prediction                  
    └── README.md

### ETL Job Orchestration using Apache Airflow 

<p align="center">
  <img width="500" height="400" src="https://github.com/arsalhuda24/MLOps-Realtime-Trajectory-Prediction/blob/master/images/airflow_jobs.jpg">
</p>



## Model Development 

<p align="center">
  <img width="800" height="350" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/models.png">
</p>

Ubuntu 20.04 python 3.8


We develop a LSTM autoencoder model. The historical motion is encoded into a low dimentional subspace through LSTM encoder and future motion is predicted by a LSTM decoder layer. 

- LSTM Autoencoder 
- Social GAN (TODO) 
- Spatio Temporal Graph Neural Network- STGCNN (TODO)

## Deployment 
- Flask App 

## Monitoring 
<p align="center">
  <img width="800" height="350" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/evalute.jpg">
</p>


The effectiveness of trajectory prediciton models is measured by ADE/FDE (meters). Lower the better. These metrics are monitored in production and re-training triggers are initiated when ADE/FDE rises above a specific threshold. 

## Machine Learning Infrastructure in AWS 

<!--![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/MLOps.png) -->
![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/ML_infrastructure.png)


## ML Infrastructure Lifecycle 

The goal is to scale the on-premise infrastructure using AWS cloud. The ETL pipleline will be deployed on EC2 instances. The heavy lifting object detection and tracking is still done locally (not to rake up cloud bill). 

#####	Data Ingestion 

- Object detecion and tracking is done locally and agent coordinates are transmitted to AWS via Kinesis streams and firehose.
- The raw data is stored eventually in S3 bucket. This completes the data ingestion procedure. 

#####	ETL Pipleline   

- Extract trajectories (sequences) with frame and pedestrian information and store in another table

#### Experimentation and Prototyping 

<!-- ![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/ML-Dev.png) -->

<img src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/ML-Dev.png" width="950" height="500">


#####	Data Selection 

- The goal is to observe pedestrian motion for (3.2 secs) and forecast their future motion (4.8sec)
- The data to be selected is pedestrian historical coordinates which form their observed trajectory
- We also want to ask a question how many days worth of pedestrian data we want to train on.
  - will depend on compute resources 

##### Model Development 

- Since its a seq-seq learning problem. Pick a baseline model such as LSTM and train. 
- Input : Past/historical sequence of pedestrian coordinates (observed trajectory)
- Output : Future sequence of pedestrian coordinates (predicted trajectory)

##### Model Validation 



##### Model Evaluation 

- Train/Test Split 
- K-fold cross validation 

After experimentation and prototyping the model goes into continuous training (CT) pipeline. Whether you use code-first, low-code, or no-code tools to build continuous training pipelines, the development artifacts, including source code and configurations, must be version controlled (for example, using Git-based source control systems)

#### Training Pipeline Setup 

<img src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/images/CICD-Training-Pipeline.png" width="950" height="500">


The training pipeline will have following workflow stages before it goes into production 

- Continuous Intergration (CI)
  - In the CI stage, the source code is unit-tested, and the training pipeline is built and integration-tested. Any artifacts that are created by the build are stored in an artifact repository.
- Continuous Delivery (CD)
  - In the CD stage, the tested training pipeline artifacts are deployed to a target environment, where the pipeline goes through end-to-end testing before being released to the production environment. 
- Production Environment  
  - The newly deployed training pipeline is smoke-tested. If the new pipeline fails to produce a deployable model, the model serving system can fall back to the model that was produced by the current training pipeline.

#### Continuous Training (CT) Pipeline 

To improve the model performance we want to ensure that the model is continuously trained as new pedestrians come into the scene and their trajectories become available. 








<!--- -- ### References 
- [A Scalable Deep Learning Rest API (Keras + Flask + Redis)](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)

- [Multi Object Detection and Tracking](https://github.com/cfotache/pytorch_objectdetecttrack)

- [YOLO5_DeepSort pedestrian detection and tracking](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

- [Training YOLOv5 on custom dataset](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)


- [Pedestrian Tracking on RTSP camera feed using pyTorch](https://medium.com/natix-io/real-time-pedestrian-tracking-service-for-surveillance-cameras-using-pytorch-and-flask-6bc9810a4cb8)

- [Python script to Cache SQL data into Redis](https://clasense4.wordpress.com/2012/07/29/python-redis-how-to-cache-python-mysql-result-using-redis/)

- [Apache Airflow introduction to manage multiple workflows](https://www.youtube.com/watch?v=2v9AKewyUEo)

- [Building ML pipelines with Airflow for end users](https://medium.com/programming-soda/apache-airflow%E3%81%A7%E3%82%A8%E3%83%B3%E3%83%89%E3%83%A6%E3%83%BC%E3%82%B6%E3%83%BC%E3%81%AE%E3%81%9F%E3%82%81%E3%81%AE%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%91%E3%82%A4%E3%83%97%E3%83%A9%E3%82%A4%E3%83%B3%E3%82%92%E6%A7%8B%E7%AF%89%E3%81%99%E3%82%8B-part3-c37ec8820033) 

- [Scalable, secure, and cost-optimized Toyota Connected Data Lake](https://aws.amazon.com/blogs/big-data/enhancing-customer-safety-by-leveraging-the-scalable-secure-and-cost-optimized-toyota-connected-data-lake/)

- [Continious Training ML pipeline in AWS](https://github.com/dylan-tong-aws/aws-serverless-ml-pipeline) 

- [MLOPS (Revisiting google paper)](https://medium.com/slalom-data-analytics/mlops-part-1-assessing-machine-learning-maturity-88e9cb05eca9) 
- [distributed video streaming pipeline using KAFKA](https://towardsdatascience.com/kafka-in-action-building-a-distributed-multi-video-processing-pipeline-with-python-and-confluent-9f133858f5a0)>


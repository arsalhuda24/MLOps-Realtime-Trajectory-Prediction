# Real-Time Trajectory Prediction System
## System Design
<p align="center">
  <img width="1000" height="250" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/system_design.jpg">
</p>
The system will have following componenets:

- ETL Pipline  
- Model Development (LSTM, GAN, VAE)
- Deployment 
- Realtime Inference
- Monitoring 
- Retrain  

## ETL Pipeline 
<p align="center">
  <img width="1000" height="200" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/ETL.jpg">
</p>

### Data Ingestion (Extract)
Video frames are read from RTSP live video feed using OpenCV
#### Object Detection and Tracking (Yolo5 + deepsort)

<!-- <p align="center">
  <img width="560" height="250" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/detection_yolo5_deepsort.gif">
</p> -->


<p align="center">
  <img width="560" height="250" src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/bev_trajs.gif">
</p>



## Machine Learning Infrastructure for Real-Time Pedestrian Trajectory Prediction

<!--![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/MLOps.png) -->
![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/ML_infrastructure.png)


## ML Infrastructure Lifecycle 

#####	Data Collection

- Run object detection/tracking algorithms (Yolo etc) to detect and track pedestrians from live video feed.
- Extract pedestrian bounding box's centroids. This serves as (x,y) coordinates of peds in a given frame
- Save coordinates and frame information in a Database or a storage service (S3) 
- Ensure realtime update of pedestrian coordinates in data storage 

#####	Data Transformation  

- Extract trajectories (sequences) with frame and pedestrian information and store in another table

#### Experimentation and Prototyping 

<!-- ![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/ML-Dev.png) -->

<img src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/ML-Dev.png" width="950" height="500">


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

<img src="https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/CICD-Training-Pipeline.png" width="950" height="500">


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


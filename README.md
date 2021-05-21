# Real-Time Trajectory Prediction Deployment

![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/Trajectory_Prediction_AWS.bmp)


## ML Infrastructure Lifecycle 

####	Data Management 

#####	Data Collection

- Run object detection/tracking algorithms (Yolo etc) to detect and track pedestrians from live video feed.
- Extract pedestrian bounding box's centroids. This serves as (x,y) coordinates of peds in a given frame
- Save coordinates and frame information in a Database or a storage service (S3) 
- Ensure realtime update of pedestrian coordinates in data storage 

#####	Data Transformation  

- Extract trajectories (sequences) with frame and pedestrian information and store in another table

#### Experimentation and Prototyping 

![Trajectory](https://github.com/arsalhuda24/Realtime-Trajectory-Prediction-AWS/blob/master/ML-Dev.png)


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

The training pipeline will have following workflow stages before it goes into production 

- Continuous Intergration (CI)
  - In the CI stage, the source code is unit-tested, and the training pipeline is built and integration-tested. Any artifacts that are created by the build are stored in an artifact repository.
- Continuous Delivery (CD)
  - In the CD stage, the tested training pipeline artifacts are deployed to a target environment, where the pipeline goes through end-to-end testing before being released to the production environment. 
- Production Environment  
  - The newly deployed training pipeline is smoke-tested. If the new pipeline fails to produce a deployable model, the model serving system can fall back to the model that was produced by the current training pipeline.

#### Continuous Training (CT) Pipeline 

To improve the model performance we want to ensure that the model is continuously trained as new pedestrians come into the scene and their trajectories become available. 








### References 
A Scalable Deep Learning Rest API (Keras + Flask + Redis)
https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/

# Driver-Behavior-Detection-CNN-Project3
In this repository, In this repository, projects related to AI, computer vision, ML, DL, machine vision, RL, CNN, ANN, RNN, NNs, robotics, NLP, LLMs, and the Transformers course will be implemented.

<img width="2700" height="1519" alt="1" src="https://github.com/user-attachments/assets/5e2e961c-47b1-4192-8472-64339a4cb42c" />
A Brief Explanation
¶
linkcode
•	1. What is the dataset about?
	This dataset is about car driver behavior detection. What does it mean? Driver behavior detection is a field of research that aims to identify and analyze the actions of drivers while they are driving. In this dataset we have 5 classes: 1- Safe driving, 2- Talking on phone, 3- Texting on phone, 4- Turning and 5- Other activities
•	2. Which structures?
	There are several types of CNN structures that are used in deep learning-based image analysis. In this project, I use three of them. AlexNet, VGGNet and ResNet. These are just some of the most popular CNN structures used in image analysis tasks. Each structure in CNN has its own unique features and advantages and because of that, if we want to use CNN for a case, we have to use more than one structure for comparing which of these structures are better fitting to our images.

About this project
¶
linkcode
•	1. Why CNN?
	CNN (Convolutional Neural Network) is a deep learning-based method that is used for driver behavior detection. CNN is a type of neural network that is designed to recognize patterns in images and videos. CNNs are particularly useful for image classification tasks because they can automatically learn to detect features such as edges, corners, and shapes in images. In driver behavior detection, CNNs are used to analyze video footage of drivers and identify patterns of behavior that may indicate unsafe driving practices.
•	2. Importance of this topic
	Data behavior detection is important because it helps to identify potential cybersecurity threats by tracking user behavior and data access activities. By analyzing both user behavior and data access activities, a behavior analytics tool can create a contextual behavior baseline to help discern behaviors that are normal from those that are not and accurately identify critical data threats.
•	Note
	I just want to show you the performance of CNN on this topic and It's all about the structure of AlexNet, VGGNet and ResNet. In next projects I will consider the other parts of this project that I mentioned on the recommendation part (Part 10).

Import Libraries
¶
In [1]:
linkcode
import warnings
warnings.filterwarnings("ignore")
from numpy import asarray
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import glob
import os 
import random
import subprocess
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.patches import Rectangle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

Import Dataset
¶
linkcode
•	Calling the images from directory
	There is 5 differente folder for each class. We have to call them to the different variables. For this purpose, I create 5 empty lists and with some for loop for iterating in folders with os.listdir. After that we can use a if for add images with png or jpg formats. You can see these methods in the cell below.
In [2]:
linkcode
image_list_other = []
image_list_safe = []
image_list_talking = []
image_list_text = []
image_list_turn = []

for other in os.listdir("/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities"): 
    if other.endswith(".png") or other.endswith(".jpg"): 
        image_list_other.append(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities",
                                             other))
        print(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities", other))
        
for safe in os.listdir("/kaggle/input/revitsone-5class/Revitsone-5classes/safe_driving"): 
    if safe.endswith(".png") or safe.endswith(".jpg"): 
        image_list_safe.append(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/safe_driving", 
                                            safe))
        print(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/safe_driving",
                           safe))
        
for talking in os.listdir("/kaggle/input/revitsone-5class/Revitsone-5classes/talking_phone"): 
    if talking.endswith(".png") or talking.endswith(".jpg"): 
        image_list_talking.append(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/talking_phone", 
                                               talking))
        print(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/talking_phone",
                           talking))

for text in os.listdir("/kaggle/input/revitsone-5class/Revitsone-5classes/texting_phone"): 
    if text.endswith(".png") or text.endswith(".jpg"): 
        image_list_text.append(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/texting_phone",
                                            text))
        print(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/texting_phone",
                           text))
        
for turn in os.listdir("/kaggle/input/revitsone-5class/Revitsone-5classes/turning"): 
    if turn.endswith(".png") or turn.endswith(".jpg"): 
        image_list_turn.append(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/turning",
                                            turn))
        print(os.path.join("/kaggle/input/revitsone-5class/Revitsone-5classes/turning", 
                           turn))
    


/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25688.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10731.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19798.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4207.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24816.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5355.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4541.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6690.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3114.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12714.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17534.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12892.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24765.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8327.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_26296.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7433.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11421.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5451.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19530.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20458.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2523.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11449.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24646.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16498.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13513.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6644.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_796.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15589.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1133.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23917.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13214.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10516.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2553.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3533.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24749.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6672.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16321.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12624.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23170.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3717.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15278.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4734.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20809.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17178.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1424.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3779.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11654.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11675.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8759.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2417-06-34.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-33-00.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7522.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8389.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1071.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18091.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10204.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_650.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23688.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13908.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3940.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8521.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8649.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5886.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2454.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14780.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23275.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8026.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1825.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6334.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19443.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14778.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25540.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20846.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17696.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14227.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22327.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22654.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14687.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12019.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2693.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6171.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3318.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10102.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16788.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13605.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7101.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8619.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13547.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16302.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24715.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7783.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13996.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8431.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1386.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7847.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8076.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12046.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13918.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18541.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_27433.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14344.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7186.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2130.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3945.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4998.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_212.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9676.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11990.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11239.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2417-26-36.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15381.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3632.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16688.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11528.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2090.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20299.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15203.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21433.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18894.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15100.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_27307.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8852.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4664.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5586.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15562.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1605.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16079.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11529.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1125.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22049.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13012.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16661.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12539.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13807.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3700.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21310.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20968.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12231.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13545.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10052.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4128.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15446.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8497.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9098.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11163.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8142.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2338.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20624.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-50-31.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20379.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_659.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17016.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16999.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12792.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3131.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1012.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8495.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20636.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19279.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2116.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5361.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12575.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2289.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11642.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20708.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18028.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5104.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12607.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12910.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16339.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13786.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19376.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2539.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12896.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19323.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5335.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11718.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3965.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12638.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6795.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24021.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18372.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6151.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17104.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20780.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22266.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10147.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2006.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1786.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14491.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11398.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7325.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18910.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14560.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11523.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4158.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18356.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16049.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6838.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5659.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4173.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25998.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8520.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8605.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2372.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5515.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16904.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17643.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8939.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13516.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25774.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6475.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7935.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6487.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17661.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14290.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12041.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_138.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5316.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9596.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18016.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8779.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9159.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11197.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2417-31-19.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-06-20.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15011.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5065.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_457.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23640.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19818.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18866.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7407.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1702.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16164.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12644.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14084.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17064.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_247.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4810.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_26585.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25239.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2412.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-34-14.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6637.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20688.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_444.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14333.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21298.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4238.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3712.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19979.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12978.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12293.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12829.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_763.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15947.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14958.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23477.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11309.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15107.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8131.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2373.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5783.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9874.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_26378.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5482.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23948.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_579.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24413.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17449.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11871.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17137.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11502.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5749.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8456.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12548.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23591.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9332.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-06-59.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18178.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20775.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6919.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3027.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23760.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21790.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25846.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7237.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_632.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16657.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18354.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22871.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25530.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7188.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-33-23.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4102.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7216.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1761.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22446.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15564.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25969.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3813.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13202.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10588.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2876.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10847.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10560.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11324.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6803.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8432.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18907.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4720.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21242.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18669.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_274.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20120.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16148.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3182.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11628.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12704.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18881.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14801.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7568.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11472.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12338.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17632.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15331.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17827.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2723.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25892.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9104.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17647.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8223.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15583.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9330.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13371.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6816.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3705.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25738.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15707.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23683.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15019.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25178.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-33-52.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7128.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2370.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2025.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11769.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15792.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_895.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10093.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18492.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1554.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14750.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4597.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20981.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13204.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_28073.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12419.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1442.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11182.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_24134.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5839.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5837.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5177.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_23876.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7974.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_125.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9164.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21946.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17633.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8564.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11964.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12535.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14240.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3430.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9649.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10232.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25438.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18208.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1790.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14125.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15962.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20952.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18111.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14932.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6285.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12635.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8482.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_118.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9728.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16828.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20358.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2417-07-07.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20956.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11050.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3400.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19543.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/2019-04-2416-46-52.png
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_27549.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8976.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13970.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6759.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17006.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14168.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12000.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22941.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11726.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10046.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4530.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16171.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13438.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4797.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10318.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19238.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19854.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21217.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10359.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12151.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19071.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16374.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20614.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5624.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11621.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10202.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18737.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_27683.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17950.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6292.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12188.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22171.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15777.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19126.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_5271.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21153.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21846.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9276.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9165.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11972.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22097.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16537.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_27993.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21871.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9471.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_9578.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_16448.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_26184.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7648.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20770.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2398.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8735.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10422.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4062.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2339.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_12370.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7136.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6845.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4658.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17948.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18411.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_11417.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6666.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_14974.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3834.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_25494.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_887.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_17273.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2409.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10473.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1766.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6089.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18000.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_19683.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_15312.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_21297.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_2421.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_1821.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_6628.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_10133.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_8407.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22553.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_18908.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_3629.jpg
/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_661.jpg


Pre Modeling
¶
linkcode
•	1. Delete unreadable images
	Sometimes we have some unreadable images that machine cannot read these images. There are two ways for handling these images. 1- Finding these images on our own way and exploring in the dataset if the dataset has little images (like this case). 2- We can create some functions that allow machine to iterate in images for finding out that which images are readable or not.
In [3]:
linkcode
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_79.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_4664.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_7973.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13318.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13396.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13541.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_13625.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_20398.jpg')
image_list_other.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/other_activities/img_22266.jpg')

image_list_turn.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/turning/img_8771.jpg')
image_list_turn.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/turning/img_62337.jpg')
image_list_turn.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/turning/img_67523.jpg')
image_list_turn.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/turning/img_70552.jpg')
image_list_turn.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/turning/img_84605.jpg')
image_list_turn.remove('/kaggle/input/revitsone-5class/Revitsone-5classes/turning/img_101434.jpg')
In [4]:
linkcode
font = {'family':'Times New Roman','color':'#1f211f'}
background_color = '#fab72f'

•	2. Plot random images from each class
	This part is for better understanding the images that we have.
•	plt.figure(1, figsize=(15, 9))
•	plt.axis('off')
•	n = 0
•	for i in range(4):
•	    
•	    n += 1
•	    random_img = random.choice(image_list_talking)
•	    imgs = imread(random_img)
•	    plt.suptitle("Random images of people who talk with their phone",
•	                 fontdict = font, fontsize=25
•	                 ,backgroundcolor= background_color)
•	    plt.subplot(2,2,n)
•	    plt.imshow(imgs)
•	
•	plt.show()


<img width="1110" height="826" alt="__results___18_0" src="https://github.com/user-attachments/assets/82a4ca68-459b-4265-82f6-1b787625ce5d" />

plt.figure(1, figsize=(15, 9))
plt.axis('off')
n = 0
for i in range(4):
    
    n += 1
    random_img = random.choice(image_list_text)
    imgs = imread(random_img)
    plt.suptitle("Random images of people who text with their phone",
                 fontdict = font, fontsize=25
                 ,backgroundcolor=background_color)
    plt.subplot(2,2,n)
    plt.imshow(imgs)

plt.show()


<img width="1110" height="826" alt="__results___19_0" src="https://github.com/user-attachments/assets/c9efb5c9-6598-49c4-8b56-9407bfedeb30" />



plt.figure(1, figsize=(15, 9))
plt.axis('off')
n = 0
for i in range(4):
    
    n += 1
    random_img = random.choice(image_list_turn)
    imgs = imread(random_img)
    plt.suptitle("Random images of people who turn around",
                 fontdict = font, fontsize=25
                 ,backgroundcolor=background_color)
    plt.subplot(2,2,n)
    plt.imshow(imgs)

plt.show()

<img width="1110" height="826" alt="__results___21_0" src="https://github.com/user-attachments/assets/96aa87bc-fa92-4ebf-bd96-2161116b9317" />


plt.figure(1, figsize=(15, 9))
plt.axis('off')
n = 0
for i in range(4):
    
    n += 1
    random_img = random.choice(image_list_safe)
    imgs = imread(random_img)
    plt.suptitle("Random images of people who drive safely",
                 fontdict = font, fontsize=25
                 ,backgroundcolor=background_color)
    plt.subplot(2,2,n)
    plt.imshow(imgs)

plt.show()


<img width="1110" height="826" alt="__results___22_0" src="https://github.com/user-attachments/assets/5ba495b8-4a38-4269-b289-2cf30e2457c4" />


plt.figure(1, figsize=(15, 9))
plt.axis('off')
n = 0
for i in range(4):
    
    n += 1
    random_img = random.choice(image_list_other)
    imgs = imread(random_img)
    plt.suptitle("Random images of people who are in other positions",
                 fontdict = font, fontsize=25
                 ,backgroundcolor=background_color)
    plt.subplot(2,2,n)
    plt.imshow(imgs)

plt.show()

•	3. Split data to train, test, prediction
	I deciede to devide dataset in this situation: 1- train = 75%, 2- test = 15% and 3- valid = 5%
In [10]:
linkcode
print("Number of samples in (Class = Other) = " ,len(image_list_other))
print("Number of samples in (Class = Safe Driving) = " ,len(image_list_safe))
print("Number of samples in (Class = Talking Phone) = " ,len(image_list_talking))
print("Number of samples in (Class = Texting Phone) = " ,len(image_list_text))
print("Number of samples in (Class = Turning) = " ,len(image_list_turn))
Number of samples in (Class = Other) =  2119
Number of samples in (Class = Safe Driving) =  2203
Number of samples in (Class = Talking Phone) =  2169
Number of samples in (Class = Texting Phone) =  2203
Number of samples in (Class = Turning) =  2057

print(.75*len(image_list_other) , .2*len(image_list_other) ,.05*len(image_list_other))
print(.75*len(image_list_safe) , .2*len(image_list_safe) ,.05*len(image_list_safe))
print(.75*len(image_list_talking) , .2*len(image_list_talking) ,.05*len(image_list_talking))
print(.75*len(image_list_text) , .2*len(image_list_text) ,.05*len(image_list_text))
print(.75*len(image_list_turn) , .2*len(image_list_turn) ,.05*len(image_list_turn))
1589.25 423.8 105.95
1652.25 440.6 110.15
1626.75 433.8 108.45
1652.25 440.6 110.15
1542.75 411.40000000000003 102.85000000000001


print("Train","Test", "Valid")

train_other = image_list_other[:1589]
test_other = image_list_other[1589:2012]
valid_other = image_list_other[2012:]

print (len(train_other), len(test_other), len(valid_other))

train_safe = image_list_safe[:1652]
test_safe = image_list_safe[1652:2092]
valid_safe = image_list_safe[2092:]

print (len(train_safe), len(test_safe), len(valid_safe))

train_talking = image_list_talking[:1626]
test_talking = image_list_talking[1626:2059]
valid_talking = image_list_talking[2059:]

print (len(train_talking), len(test_talking), len(valid_talking))

train_text = image_list_text[:1652]
test_text = image_list_text[1652:2092]
valid_text = image_list_text[2092:]
print (len(train_text), len(test_text), len(valid_text))

train_turn = image_list_turn[:1547]
test_turn = image_list_turn[1547:1959]
valid_turn = image_list_turn[1959:]

print (len(train_turn), len(test_turn), len(valid_turn))
Train Test Valid
1589 423 107
1652 440 111
1626 433 110
1652 440 111
1547 412 98
linkcode

•	4. Create dataframes for train, test, prediction
	If you don't want to use directly from the directories and you want to create some lists of images, you can turn the format of data to dataframes and use flow from dataframe method that we see in the next cells. For this approach, create a label column for use it as a tag or label
In [13]:
linkcode
train_other_df = pd.DataFrame({'image':train_other, 'label':'Other'})
train_safe_df = pd.DataFrame({'image':train_safe, 'label':'Safe'})
train_talking_df = pd.DataFrame({'image':train_talking, 'label':'Talk'})
train_text_df = pd.DataFrame({'image':train_text, 'label':'Text'})
train_turn_df = pd.DataFrame({'image':train_turn, 'label':'Turn'})
In [14]:
linkcode
test_other_df = pd.DataFrame({'image':test_other, 'label':'Other'})
test_safe_df = pd.DataFrame({'image':test_safe, 'label':'Safe'})
test_talking_df = pd.DataFrame({'image':test_talking, 'label':'Talk'})
test_text_df = pd.DataFrame({'image':test_text, 'label':'Text'})
test_turn_df = pd.DataFrame({'image':test_turn, 'label':'Turn'})

valid_other_df = pd.DataFrame({'image':valid_other, 'label':'Other'})
valid_safe_df = pd.DataFrame({'image':valid_safe, 'label':'Safe'})
valid_talking_df = pd.DataFrame({'image':valid_talking, 'label':'Talk'})
valid_text_df = pd.DataFrame({'image':valid_text, 'label':'Text'})
valid_turn_df = pd.DataFrame({'image':valid_turn, 'label':'Turn'})
In [16]:
linkcode
train_df = pd.concat([train_other_df, train_safe_df, train_talking_df, train_text_df, train_turn_df])
test_df = pd.concat([test_other_df, test_safe_df, test_talking_df, test_text_df, test_turn_df])
val_df = pd.concat([valid_other_df, valid_safe_df, valid_talking_df, valid_text_df, valid_turn_df])
In [17]:
linkcode
train_df.head()

	image	label
0	/kaggle/input/revitsone-5class/Revitsone-5clas...	Other
1	/kaggle/input/revitsone-5class/Revitsone-5clas...	Other
2	/kaggle/input/revitsone-5class/Revitsone-5clas...	Other
3	/kaggle/input/revitsone-5class/Revitsone-5clas...	Other
4	/kaggle/input/revitsone-5class/Revitsone-5clas...	Other
In [18]:
linkcode
print("Number of rows in train dataframe is: ", len(train_df))
print("Number of rows in test dataframe is: ", len(test_df))
print("Number of rows in val dataframe is: ", len(val_df))
Number of rows in train dataframe is:  8066
Number of rows in test dataframe is:  2148
Number of rows in val dataframe is:  537
•	5. Calculate the ratio of images
	It's really important in some cases for knowing the ratio of images that we use. We can use cv2.imread for this part.
In [19]:
linkcode
random_img_height = random.choice(train_other)
In [20]:
linkcode
image= cv2.imread(random_img_height)

height, width= image.shape[:2]

print("The height is ", height)

print("The width is ", width)
The height is  480
The width is  640


•	6. Define hyperparameters
	Some of the hyperparameters should be considering before the model goes to start. I set Batch size as 64, ofcourse you can use 32 or any other popular numbers or you can use some functions for checking the best number of that but it costs lots of run time and I just want to pick some experimental value. Also for AlexNet, ResNet and VGGNet its popular to use height and width as a 240*240 format.
In [21]:
linkcode
Batch_size = 64
Img_height = 240
Img_width = 240
linkcode
•	7. Rescale the images
	Its really important to rescale the images and put all of the images in a same shape for input layer of model.
In [22]:
linkcode
trainGenerator = ImageDataGenerator(rescale=1./255.)
valGenerator = ImageDataGenerator(rescale=1./255.)
testGenerator = ImageDataGenerator(rescale=1./255.)

•	8. Input Model
	Now we can use flow from dataframe that mentioned before. It helps to call images from some dataframes and mention the label as target.
In [23]:
linkcode
trainDataset = trainGenerator.flow_from_dataframe(
  dataframe=train_df,
  class_mode="categorical",
  x_col="image",
  y_col="label",
  batch_size=Batch_size,
  seed=42,
  shuffle=True,
  target_size=(Img_height,Img_width) #set the height and width of the images
)

testDataset = testGenerator.flow_from_dataframe(
  dataframe=test_df,
  class_mode='categorical',
  x_col="image",
  y_col="label",
  batch_size=Batch_size,
  seed=42,
shuffle=True,
  target_size=(Img_height,Img_width)
)

valDataset = valGenerator.flow_from_dataframe(
  dataframe=val_df,
  class_mode='categorical',
  x_col="image",
  y_col="label",
  batch_size=Batch_size,
  seed=42,
  shuffle=True,
  target_size=(Img_height,Img_width)
)
Found 8066 validated image filenames belonging to 5 classes.
Found 2148 validated image filenames belonging to 5 classes.
Found 537 validated image filenames belonging to 5 classes.


AlexNet
¶
linkcode
•	1. What is AlexNet?
	AlexNet is a convolutional neural network (CNN) architecture that was introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. It is used primarily for image recognition and classification tasks. AlexNet was the winner of the ImageNet Large Scale Visual Recognition Challenge in 2012, which marked a breakthrough in deep learning. The network contains eight layers; the first five are convolutional layers, some of them followed by max-pooling layers, and the last three are fully connected layers. The network is split into two copies, each run on one GPU. AlexNet uses ReLU activation functions and dropout regularization to prevent overfitting. The architecture of AlexNet has inspired many other CNN architectures that have been developed since its introduction.
linkcode
•	2. Model Structure
	You can see the structure of AlexNet in the cell below.

def AlexNet():
    inp = layers.Input((240, 240, 3))
    x = layers.Conv2D(96, 11, 4, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = layers.Conv2D(256, 5, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = layers.Conv2D(384, 3, 1, activation='relu')(x)
    x = layers.Conv2D(384, 3, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, 1, activation='relu')(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5, activation='softmax')(x)

    model_Alex = models.Model(inputs=inp, outputs=x)

    return model_Alex
model_Alex = AlexNet()
model_Alex.summary()
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 240, 240, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 58, 58, 96)        34944     
                                                                 
 batch_normalization (BatchN  (None, 58, 58, 96)       384       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 28, 28, 96)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 256)       614656    
                                                                 
 batch_normalization_1 (Batc  (None, 24, 24, 256)      1024      
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 11, 11, 256)      0         
2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 9, 9, 384)         885120    
                                                                 
 conv2d_3 (Conv2D)           (None, 7, 7, 384)         1327488   
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 256)         884992    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 2, 2, 256)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1024)              0         
                                                                 
 dense (Dense)               (None, 4096)              4198400   
                                                                 
 dropout (Dropout)           (None, 4096)              0         
                                                                 
 dense_1 (Dense)             (None, 4096)              16781312
dropout_1 (Dropout)         (None, 4096)              0         
                                                                 
 dense_2 (Dense)             (None, 5)                 20485     
                                                                 
=================================================================
Total params: 24,748,805
Trainable params: 24,748,101
Non-trainable params: 704
_________________________________________________________________
In [25]:
linkcode
tf.keras.utils.plot_model(
    model_Alex,
    to_file='alex_model.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    show_layer_activations=True,
    dpi=100
)

<img width="376" height="1350" alt="image" src="https://github.com/user-attachments/assets/362f8390-1885-4e93-8c91-2d41fd8a4605" />




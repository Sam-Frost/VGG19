import numpy as np
import pandas as pd 
import pickle
import cv2
from os import listdir
from tensorflow import keras 
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense, ReLU, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import tensorflow

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from tensorflow.keras.applications.vgg16 import VGG16

default_image_size = tuple((224,224))
directory_root = '/kaggle/input/covid-data'

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
import random

N_IMAGES = 3059
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    plant_disease_folder_list = listdir(f"{directory_root}")
    plant_healthy_folder_list = listdir(f"{directory_root}")
    for directory in plant_disease_folder_list :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)
    for directory in plant_healthy_folder_list :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)  

    for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
            #print(plant_disease_image_list)
    for plant_healthy_folder in plant_healthy_folder_list:
            print(f"[INFO] Processing {plant_healthy_folder} ...")
            plant_healthy_image_list = listdir(f"{directory_root}/{plant_healthy_folder}/")
                
    for image in plant_disease_image_list[:N_IMAGES]:
                image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(random.shuffle(convert_image_to_array(image_directory)))
                    label_list.append(plant_disease_folder)
    for image in plant_healthy_image_list[:N_IMAGES]:
                image_directory = f"{directory_root}/{plant_healthy_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(random.shuffle(convert_image_to_array(image_directory)))
                    label_list.append(plant_healthy_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
directory_root = datagen.flow_from_directory('/kaggle/input/covid-data', class_mode='categorical')

image_array_list, label_list = [], []
totalImage = 0
root_dir = listdir('/kaggle/input/covid-data')
for image_folder in root_dir:
    image_list = listdir(f"/kaggle/input/covid-data/{image_folder}")
    for image in image_list:
        image_name = f"/kaggle/input/covid-data/{image_folder}/{image}"
        if isinstance (image_name,str)==True:
            if image_name.endswith(".jpg") == True or image_name.endswith(".JPG") == True or image_name.endswith(".jpeg") == True or image_name.endswith(".JPEG") == True or image_name.endswith(".PNG") == True or image_name.endswith(".png") == True:
                    image_array_list.append(convert_image_to_array(image_name))
                    label_list.append(image_folder)
                    totalImage=totalImage+1
print(len(label_list),len(np.unique(np.asarray(label_list))))
from sys import getsizeof
getsizeof(image_list)

print(f"total  number of images",totalImage)
n_classes = len(np.unique(np.asarray(label_list)))
print("total number of classes",n_classes)
#print(plant_disease_folder)

np_image_array_list = np.array(image_array_list, dtype=np.float32) / 255.0
print()
image_len = len(image_array_list)
print(f"Total number of images: {image_len}")

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('plant_disease_label_transform.pkl', 'wb'))
pickle.dump(label_binarizer,open('plant_healthy_label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
print("total number of classes: ", n_classes)
print(label_binarizer.classes_)

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_array_list, image_labels, test_size=0.3, random_state = 42)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

VGG19 = tensorflow.keras.applications.vgg19.VGG19(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224,224,3),
    pooling="max"
)

x= VGG19.layers[-1].output 
nmodel = tensorflow.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
model = tensorflow.keras.Model(inputs = VGG19.input, outputs = nmodel)
#model.add(Dense(units=2, activation="softmax"))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)
loss, accuracy = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype(int)
y_pred_classes = [np.argmax(element) for element in y_pred]
y_pred_classes = to_categorical(y_pred_classes)
f1score = f1_score(y_test, y_pred_classes, average='weighted')
print("Classification Report: \n", classification_report(y_test, y_pred_classes))
model.save('/kaggle/working/vgg19_model.h5')



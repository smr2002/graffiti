import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16


def process(data):
    if not os.path.exists(os.path.join("graffiti_data",data)):
        raise FileNotFoundError("No such data folder. Make sure you have the correct working directory, and  try 'train', 'test', or 'valid'.")
    files = sorted([file for file in \
                    os.listdir(os.path.join("graffiti_data",data))\
                    if file[-4:] == ".jpg"])
    X = [load_img(os.path.join("graffiti_data",data,im)) for im in files]
    X = [img_to_array(im) for im in X]
    X = [im.reshape((1, im.shape[0], im.shape[1], im.shape[2])) for im in X]
    X = np.array([preprocess_input(im) for im in X])
    classes = pd.read_csv(os.path.join('graffiti_data',data,'_classes.csv'))\
    .sort_values("filename")
    classes.insert(1,"pixvals",list(X))
    classes.reset_index(drop=True,inplace=True)
    new_path = os.path.join("graffiti_data",f"{data}_data_full.csv")
    if os.path.exists(new_path):
        os.remove(new_path)
    classes.to_csv(new_path,index=False)
    return classes
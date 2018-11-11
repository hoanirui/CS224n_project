from LSTM_model import LSTM_Config, LSTM_Model
import numpy as np
import tensorflow as tf

import pickle as cPickle
import os
import time
import json
import random
import shutil

from utilities import train_data_iterator, detokenize_caption, evaluate_captions
from utilities import plot_performance, log

def main():
    dir = "coco/data/img_features_attention/"
    # create a list of the paths to all val imgs:
    paths = [dir + file_name for file_name in os.listdir(dir)]

    img_id_2_feature_array = {}

    for step, path in enumerate(paths):
        if step % 1000 == 0:
            print(step)
            log(str(step))
        file_name = path.split("/")[3]
        if file_name not in ["transform_params", "-1"]:
            img_id = int(file_name)
            feature_array = cPickle.load(open(path, "rb"))
            img_id_2_feature_array[img_id] = feature_array

    cPickle.dump(img_id_2_feature_array,
                 open("coco/data/img_id_2_feature_array", "wb"))

if __name__ == '__main__':
    main()

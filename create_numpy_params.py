from LSTM_model import LSTM_Config, LSTM_Model
import numpy as np
import tensorflow as tf

import pickle as cPickle
import os
import time
import json
import random

from utilities import train_data_iterator, detokenize_caption, evaluate_captions
from utilities import plot_performance, log

def main():
    # create a config object:
    config = LSTM_Config()
    # get the pretrained embeddings matrix:
    GloVe_embeddings = cPickle.load(open("coco/data/embeddings_matrix", "rb"))
    GloVe_embeddings = GloVe_embeddings.astype(np.float32)
    # create an LSTM model object:
    model = LSTM_Model(config, GloVe_embeddings)

    # create the saver:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore all model variables:
        params_dir = "weights/model-41"
        saver.restore(sess, "models/LSTMs/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1/%s" % params_dir)

        # get the restored W_img and b_img:
        with tf.variable_scope("img_transform", reuse=True):
            W_img = tf.get_variable("W_img")
            b_img = tf.get_variable("b_img")

            W_img = sess.run(W_img)
            b_img = sess.run(b_img)

            transform_params = {}
            transform_params["W_img"] = W_img
            transform_params["b_img"] = b_img
            cPickle.dump(transform_params, open("coco/data/img_features_attention/transform_params/numpy_params", "wb"))

if __name__ == '__main__':
    main()

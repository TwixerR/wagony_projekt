
import numpy as np
import os
import re

import prep
import net
import process

import tensorflow as tf
import pandas as pd
from tensorflow import keras

global wagony_path

DEBUG = False
STAGE = "TEST"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # specify external dir path
    wagony_path = r"E:\2021-wagony-final (1)\wagony"
    # filter only image containing dirs based on name pattern matching
    dirs = list(filter(lambda x: x if re.match(r"\d+", x) is not None else None, os.listdir(wagony_path)))
    # declare structure storing all dir:filename tuples
    images_map = dict()
    # populate data structure
    # images_map contains keys(relevant directory names under wagony_path) and values(extensionless image file names)
    for dir in dirs:
        images_map[dir] = {c[:-5] for c in os.listdir(f"{wagony_path}\\{dir}") if
                           re.match(r"\d+\-\d+.\.jpeg", c) is not None}
    if STAGE == "PREPROCESS":
        for key in images_map.keys():
            for img_filename in images_map[key]:
                prep.preprocess(f"{key}\\{img_filename}", wagony_path)
    if STAGE == "NETWORK":
        hists = net.runall()
        print(len(hists))
    if STAGE == "PROD":
        # model = keras.models.load_model(net.BEST_MODEL_PATH)
        text = process.extract('20191105000000\\164-5R', wagony_path)
        print(text)
    if STAGE == "TEST":
        # create table headers list
        labels = ['A', 'TAG', 'DATE_TIME', 'val_A', 'val_B', 'val_C']
        for dir in images_map.keys():
            # read table
            table = pd.read_csv(f'{wagony_path}\\{dir}\\table.csv', sep=';', header=0, names=labels)
            # get values from column containing true labels
            truths = table['TAG'].to_numpy(dtype=str)
            # concatenate labels on dashes
            for item in truths:
                item = "".join(item.split('-'))
            extractions = []
            # sort directory image names Windows-style descending order to align table items with responses
            files = sorted(list(images_map[dir]), key=lambda x: int(x.split('-')[0]), reverse=True)
            for img_filename in files:
                text = process.extract(f"{dir}\\{img_filename}", wagony_path)
                extractions.append(text)
            # assess results
            dld("".join(t.split("-")), e)

def dld(given, target):
    # stores subproblems
    mat = np.zeros((len(given)+1, len(target)+1))
    # init table
    for i in range(len(given) + 1):
        mat[i][0] = i
    for k in range(len(target) + 1):
        mat[0][k] = k

    for i in range(1, len(given) + 1):
        for k in range(1, len(target) + 1):
            if given[i - 1] == target[k - 1]:
                mat[i][k] = mat[i-1][k-1]
            else:
                mat[i][k] = 1 + min(mat[i-1][k], mat[i][k-1], mat[i-1][k-1])

    return mat[len(given)][len(target)]


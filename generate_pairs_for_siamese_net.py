# coding: utf-8
import pandas as pd
import numpy as np
import itertools
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input dataset")
parser.add_argument("--output", help="output dataset, which will contain data that can be fed to a siamese neural network, the even rows should be fed to one of the siamese branches and the odd rows to the other branch")
args = parser.parse_args()

ds = pd.read_csv(args.input, sep=None, engine='python',  dtype={'user_id': "category"})


L = 2
list_per_ifp_id = []
counter = 0
for ifp_id in ds["ifp_id"].unique():
    cifp = ds[ds["ifp_id"] == ifp_id]
    for subset in itertools.combinations(cifp.values.tolist(), L):
        list_per_ifp_id += subset

 
d = pd.DataFrame(data = list_per_ifp_id, columns = ds.columns)
d.to_csv(args.output, index = False)


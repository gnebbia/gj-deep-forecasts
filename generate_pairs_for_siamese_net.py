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

 
df = pd.DataFrame(data = list_per_ifp_id, columns = ds.columns)

df_even =  df.iloc[::2]
df_odd = df.iloc[1::2]


df_even = df_even.rename(columns={'user_id': 'user_id_1', 'value_a': 'value_a_1', 'value_b': 'value_b_1',
                               'value_c': 'value_c_1', 'days_from_start': 'days_from_start_1',
                               'distance': 'distance_1', 'expertise':'expertise_1'})

df_odd = df_odd.rename(columns={'user_id': 'user_id_2', 'value_a': 'value_a_2', 'value_b': 'value_b_2',
                               'value_c': 'value_c_2', 'days_from_start': 'days_from_start_2',
                               'distance': 'distance_2', 'expertise':'expertise_2'})
df_odd = df_odd.drop(['ifp_id','topic_0','topic_1','topic_2','topic_3','topic_4','topic_5','a','b','c'], 1)

df_odd = df_odd.reset_index(drop=True)
df_even = df_even.reset_index(drop=True)

df_complete = pd.concat([df_even, df_odd], axis=1)
df_complete.to_csv(args.output, index=False)


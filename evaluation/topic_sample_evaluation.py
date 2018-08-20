import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gjnn.model
import gjnn.dataloader
import gjnn.dataset_preprocessing
import argparse
import logging.config
import timeit
from sklearn import preprocessing
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.config.fileConfig('conf/logging.conf')
writer = SummaryWriter("/home/derek/deep-forecasts/logs")

# create logger
logger = logging.getLogger('debug')

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input dataset", default="ds_short.csv", type=str)
parser.add_argument("--epochs", help="number of epochs in the training phase", default=50, type=int)
parser.add_argument("--siamese_size", help="number of neurons for siamese network", default=20, type=int)
parser.add_argument("--hidden_size", help="number of neurons for hidden fully connected layers", default=25, type=int)
parser.add_argument("--batch_size", help="size of the batch to use in the training phase", default=64, type=int)
# args = parser.parse_args()
args = parser.parse_args(
    ["--siamese_size=32", "--hidden_size=64", "--epochs=50", "--batch_size=512", "--input=ds_sample_5M.csv"])

print(args)

# shuf -n 1000000 ds_with_combinations_yr1.csv > ds_sample_1M.csv
# cat ds_sample_1M.csv >> header.csv
# mv header.csv ds_sample_1M.csv
# sed -i 's/\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,//g' das_sample_1M.csv

# Dataset Loading
logger.debug("Reading Dataset...")
print("Reading Dataset...")
x = timeit.time.time()
dataset = pd.read_csv(args.input, sep=None, engine='python', dtype={'user_id_1': "category", "user_id_2": "category"})
print("Done! It took {:.3f} seconds".format(timeit.time.time() - x))
dataset.drop(["ifp_id"], axis=1, inplace=True)

logger.info(dataset.head())

logger.debug(len(dataset))

dataset = dataset.apply(pd.to_numeric)

min_max_data_values, dataset = gjnn.dataset_preprocessing.prepare_data(dataset)

d = dataset[["topic_{}".format(x) for x in range(0,6)]]

dominant_topic = [np.argmax(d.loc[x, ]) for x in range(0, 5000000)]


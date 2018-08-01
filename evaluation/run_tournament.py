import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gjnn.model
import gjnn.dataloader
import argparse
import logging.config
import timeit
from sklearn import preprocessing
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## The problem: We need to normalize the tournament ifp's
## using the normalization scale
## from the training data.
def prepare_data_for_evaluation(dataset):

    # To modify when dataset column order will change
    features_user_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]
    features_user_2 = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22]
    topics = [0, 1, 2, 3, 4, 5]

    dataset = pd.read_csv("ds_sample_5M.csv", sep=None, engine='python',
                                dtype={'user_id_1': "category", "user_id_2": "category"})
    dataset.drop(["ifp_id"], axis=1, inplace=True)
    dataset = dataset.apply(pd.to_numeric)

    ## Normalize columns
    min_max_scaler = preprocessing.MinMaxScaler()
    topic_0_norm_map = {}
    topic_1_norm_map = {}
    topic_2_norm_map = {}
    topic_3_norm_map = {}
    topic_4_norm_map = {}
    topic_5_norm_map = {}
    expertise_1_norm_map = {}
    user_id_1_norm_map = {}
    value_a_1_norm_map = {}
    value_b_1_norm_map = {}
    value_c_1_norm_map = {}
    days_from_start_1_norm_map = {}
    a_norm_map = {}
    b_norm_map = {}
    c_norm_map = {}
    expertise_2_norm_map = {}
    user_id_2_norm_map = {}
    value_a_1_norm_map = {}
    value_b_1_norm_map = {}
    value_c_1_norm_map = {}
    days_from_start_2_norm_map = {}

    user_id_2_norm_map = {}



    ## categorical columns
    dataset["user_id_1"] = dataset["user_id_1"].fillna(0.0).astype(int)
    dataset["user_id_2"] = dataset["user_id_2"].fillna(0.0).astype(int)
    dataset["expertise_1"] = dataset["expertise_1"].fillna(0.0).astype(int)
    dataset["expertise_2"] = dataset["expertise_2"].fillna(0.0).astype(int)
    min_id = min(min(dataset["user_id_1"]), min(dataset["user_id_2"]))
    max_id = max(max(dataset["user_id_1"]), max(dataset["user_id_2"]))
    min_expertise = min(min(dataset["expertise_1"]), min(dataset["expertise_2"]))
    max_expertise = max(max(dataset["expertise_2"]), max(dataset["expertise_2"]))

    for i in dataset.columns:
        if i.startswith("distance"):
            # Don't normalize the distance columns!
            continue
        if i == "user_id_1":
            dataset["user_id_1"] = dataset["user_id_1"].transform(lambda x: (float(x) - min_id) / (max_id - min_id))
        elif i == "user_id_2":
            dataset["user_id_2"] = dataset["user_id_2"].transform(lambda x: (float(x) - min_id) / (max_id - min_id))
        elif i == "expertise_1":
            dataset["expertise_1"] = dataset["expertise_1"].transform(
                lambda x: (float(x) - min_expertise) / (max_expertise - min_expertise))
        elif i == "expertise_2":
            dataset["expertise_2"] = dataset["expertise_2"].transform(
                lambda x: (float(x) - min_expertise) / (max_expertise - min_expertise))
        else:
            x = dataset[i]  # returns a numpy array
            x_scaled = min_max_scaler.fit_transform(x)
            dataset[i] = x_scaled

    user_1 = dataset.iloc[:, features_user_1]
    user_2 = dataset.iloc[:, features_user_2]

    return dataset

def get_sample_question_data_loaders():
    data_loaders = []
    for j in range(1050, 1052): #1105):
        if j == 1079:
            continue
        filename = "ds_" + str(j) + "-0.csv"
        d = pd.read_csv(filename, sep=None, engine='python',
                              dtype={'user_id_1': "category", "user_id_2": "category"})
        d.drop(["ifp_id"], axis=1, inplace=True)
        d = d.apply(pd.to_numeric)
        d = prepare_data(d)
        test = gjnn.dataloader.Dataset(d)
        data_loaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=False))
    return data_loaders


writer = SummaryWriter("/home/derek/deep-forecasts/logs/evaluation")

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

# Dataset Loading
print("Reading Trial Questions...")
x = timeit.time.time()
data_loaders = get_sample_question_data_loaders()
print("Done! It took {:.3f} seconds".format(timeit.time.time() - x))

num_epochs = args.epochs

# Load model from disk
model = gjnn.model.SiameseNetwork(num_features_per_branch, siamese_layer_size, hidden_layer_size, output_layer_size)
model = model.to(device)
criterion = nn.modules.loss.BCEWithLogitsLoss()

model.eval()
print("Running Against out-of-sample IFIPs...")
for loader in data_loaders:
    k = 0
    competitors = {}
    for o, (oot_user_1, oot_user_2, _, _) in enumerate(loader):
        break

    total_acc = []
    total_loss = []
    for o, (oot_user_1, oot_user_2, oot_user_1_dist, oot_user_2_dist) in enumerate(loader):
        oot_user_1 = oot_user_1.to(device)
        oot_user_2 = oot_user_2.to(device)
        oot_user_1_dist = oot_user_1_dist.to(device)
        oot_user_2_dist = oot_user_2_dist.to(device)

        outputs = model(oot_user_1, oot_user_2)
        comparison = (outputs.squeeze() > 0)
        targets = (oot_user_2_dist - oot_user_1_dist > 0).type_as(outputs)
        total_loss.append(criterion(outputs.squeeze(), targets))
        total_acc.append(
            torch.mean((comparison == targets.type_as(comparison)).type_as(torch.FloatTensor())).data)
        if o % 20 == 0:
            print("Still working... {}/{}".format(o, len(loader)))

    loss = sum(total_loss) / len(total_loss)
    acc = sum(total_acc) / len(total_acc)
    writer.add_scalar('IFIP_' + str(j) + '/loss', loss, epoch + 1)
    writer.add_scalar('IFIP_' + str(j) + '/acc', acc, epoch + 1)
    print("Loss on out of sample IFIP_{} for epoch {} is: {:.4f}. Acc: {:.4f}\n".format(j, epoch + 1, loss, acc))




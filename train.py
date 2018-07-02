import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import gjnn.model
import gjnn.loss
import gjnn.dataloader
import argparse
import logging
import logging.config

logging.config.fileConfig('conf/logging.conf')


# create logger
logger = logging.getLogger('debug')

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input dataset")
args = parser.parse_args()


# Dataset Loading 
logger.debug("Reading Dataset...")
dataset = pd.read_csv(args.input, sep=None, engine='python',  dtype={'user_id_1': "category", "user_id_2":"category"})


dataset.drop(["ifp_id"], axis = 1, inplace = True)

logger.info(dataset.head())

# To modify when dataset column order will change
features_user_1 = [0,1,2,3,4,5,6,7,8,9,10,11,15]
features_user_2 = [0,1,2,3,4,5,16,17,18,19,20,21,22]
logger.debug(dataset.columns[[0,1,2,3,4,5,6,7,8,9,10,11,15]])
logger.debug(dataset.columns[[0,1,2,3,4,5,16,17,18,19,20,21,22]])

user_1 = dataset.iloc[:, features_user_1]
user_2 = dataset.iloc[:, features_user_2]

logger.debug(len(dataset))



dataset = dataset.apply(pd.to_numeric)



# The current split is 95% of data is used for training and 5% for validation of the model
train = dataset.sample(frac=0.95,random_state=200)
test = dataset.drop(train.index)


train = gjnn.dataloader.Dataset(train) 
test = gjnn.dataloader.Dataset(test)


# Some Neural Network Training Related Hyperparameters
batch_size = 64
n_iters = 1000
num_epochs = n_iters / (len(train) / batch_size)
num_epochs = int(num_epochs)






train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)



logger.debug("Dataset has been properly loaded...")

# Setting other neural network hyperparameters
hidden_layer_size = 20
siamese_layer_size = 20
output_layer_size = 1
num_features_per_branch = 13
lr = 0.01
momentum = 0.9
num_epoch = 5

logger.debug("Neural Network Hyperparameters...")
logger.debug("The number of epochs is: " + str(num_epochs))
logger.debug("The number of iterations is: " + str(n_iters))
logger.debug("The batch size is: " + str(n_iters))
logger.debug("The learning rate is: " + str(lr))
logger.debug("The number of neurons in each siamese input layer is: " + str(num_features_per_branch))
logger.debug("The number of neurons for each siamese hidden layer is: " + str(siamese_layer_size))
logger.debug("The number of neurons for the fully connected layers is: " + str(hidden_layer_size))
logger.debug("The number of neurons in the output layer is: " + str(output_layer_size))

# Check dimensions of features
model = gjnn.model.SiameseNetwork(num_features_per_branch, siamese_layer_size, hidden_layer_size, output_layer_size)
logger.debug("Model correctly initialized...")

# Initialization of the Loss Function
criterion = gjnn.loss.DistanceLoss()
logger.debug("Distance Loss Correctly Initialized...")

# At the moment we stick to a classic SGD algorithm, maybe we can change it to Adam
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
logger.debug("Optimizer Instantiated...")


iterat = 0


losses = []
for epoch in range(num_epochs):
    logger.debug("Epoch " + str(epoch))
    for i, (user_1, user_2, user_1_dist, user_2_dist) in enumerate(train_loader):
        
        # Make each batch a Torch Variable
        features_u1 = Variable(user_1)
        features_u2 = Variable(user_2)
        dist_u1 = Variable(user_1_dist)
        dist_u2 = Variable(user_2_dist)
        
        optimizer.zero_grad()
        
        # Here we have to feed data to the neural network, we insert data we want to
        # give as input to the siamese layers
        outputs = model(features_u1, features_u2)
        

        loss = criterion(user_1_dist, user_2_dist, outputs)
        
        # Keep track of loss values
        losses.append(loss)
        
        
        print("loss for i {} is equal to: {}".format(i, loss))
        
        loss.backward()
        
        optimizer.step()
        
        iterat += 1
        print(iterat)



# Printing the sequence of losses
for i in losses:
    print(i)


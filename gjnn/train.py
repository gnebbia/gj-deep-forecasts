import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import model 
import loss


# Dataset Loading 
dataset = pd.read_csv("~/ds_with_combinations.csv", sep=None, engine='python',  dtype={'user_id': "category"})

output_col_idx = 4
num_features = 12



#dataset.loc[dataset[output_col_idx]=='a', output_col_idx]=0
#dataset.loc[dataset[output_col_idx]=='b', output_col_idx]=1
#dataset.loc[dataset[output_col_idx]=='c', output_col_idx]=2
#dataset = dataset.apply(pd.to_numeric)


# The current split is 95% of data is used for training and 5% for validation of the model
train=dataset.sample(frac=0.95,random_state=200)
test=dataset.drop(train.index)

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(xtrain) / batch_size)
num_epochs = int(num_epochs)


train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

# Setting other neural network hyperparameters

hidden_layer_size = 20
lr = 0.01
num_epoch = 5 


model = Net(num_features, hidden_layer_size, num_classes)

criterion = loss.DistanceLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

iter = 0
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        print("the variable i is: " + str(i))
        print("the features are: " + str(features))
        print("the features are: " + str(labels))
        features = Variable(features.view(-1, num_features))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        
        output = model(features1, features2)
        
        distance_fc1 = 0
        distance_fc2 = 0
        
        loss = criterion(distance_fc1, distance_fc2, output)
        
        loss.backward()
        
        optimizer.step()
        
        iter += 1
        
        # we want to check the accuracy with test dataset every 500 iterations
        # we can change this number, it is just if it is too small we lose a lot of time
        # checking accuracy while if it is big, we have less answers but takes less time for the algorithm
        if iter % 500 == 0:
            # calculate accuracy
            correct = 0
            total = 0
            
            # iterate through test dataset
            for features, labels in test_loader:
                features = Variable(features.view(-1, num_features))
                
                outputs = model(features)
                # get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # total number of labels
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
            
            print("Iteration: {}. Loss: {}. Accuracy: {}".format(iter, loss.data[0], accuracy))

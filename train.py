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
np.save("gjnn/min_max_dataset_"+args.input+".npy", min_max_data_values)

# Here a custom Data Loader is used
train = gjnn.dataloader.Dataset(dataset)


batch_size = args.batch_size
num_epochs = args.epochs

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

## How do you know?
logger.debug("Dataset has been properly loaded...")

# Setting other neural network hyperparameters
hidden_layer_size = args.hidden_size
siamese_layer_size = args.siamese_size
output_layer_size = 1
num_features_per_branch = 13
lr = 0.01
momentum = 0.9

# logger.debug("The number of iterations is: " + str(n_iters))
logger.debug("Neural Network Hyperparameters...")
logger.debug("The number of epochs is: " + str(num_epochs))
logger.debug("The batch size is: " + str(batch_size))
logger.debug("The learning rate is: " + str(lr))
logger.debug("The number of neurons in each siamese input layer is: " + str(num_features_per_branch))
logger.debug("The number of neurons for each siamese hidden layer is: " + str(siamese_layer_size))
logger.debug("The number of neurons for the fully connected layers is: " + str(hidden_layer_size))
logger.debug("The number of neurons in the output layer is: " + str(output_layer_size))

# Check dimensions of features
model = gjnn.model.SiameseNetwork(num_features_per_branch, siamese_layer_size, hidden_layer_size, output_layer_size)

model = model.to(device)
logger.debug("Model correctly initialized...")

# Initialization of the Loss Function
# criterion = gjnn.loss.DistanceLoss()
criterion = nn.modules.loss.BCEWithLogitsLoss()
logger.debug("Distance Loss Correctly Initialized...")

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
logger.debug("Optimizer Instantiated...")

model.train()
count = 0
for epoch in range(num_epochs):
    print("Epoch: " + str(epoch))

    ## Training loop
    for i, (user_1, user_2, user_1_dist, user_2_dist) in enumerate(train_loader):
        user_1 = user_1.to(device)
        user_2 = user_2.to(device)
        user_1_dist = user_1_dist.to(device)
        user_2_dist = user_2_dist.to(device)

        optimizer.zero_grad()

        # Here we have to feed data to the neural network, we insert data we want to
        # give as input to the siamese layers
        outputs = model(user_1, user_2)
        comparison = (outputs.squeeze() > 0)
        targets = (user_2_dist - user_1_dist > 0).type_as(outputs)
        loss = criterion(outputs.squeeze(), targets)

        loss.backward()

        count += 1
        if i % 50 == 0:
            acc = torch.mean((comparison == targets.type_as(comparison)).type_as(torch.FloatTensor()))

            writer.add_scalar('Train/loss', loss, count)
            writer.add_scalar('Train/acc', acc, count)
            print(
            "Loss after batch {}/{} for epoch {} is: {:.4f}. Acc: {:.4f}\n".format(i + 1, len(train_loader), epoch + 1,
                                                                                   loss, acc))

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.data.cpu().numpy(), count, bins='doane')
                if value.grad is not None:
                    writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), count, bins='doane')

        optimizer.step()

    print("Epoch done! Saving model.")
    torch.save(model.state_dict(), "/home/derek/deep-forecasts/model_after_epoch_{}.pt".format(epoch+1))

    print("Running Against out-of-sample IFIPs...")
    for j in range(1050, 1105):
        ## This IFIP ID is missing.
        if j == 1079:
            continue
        filename = "ds_" + str(j) + "-0.csv"
        dataset = pd.read_csv(filename, sep=None, engine='python',
                              dtype={'user_id_1': "category", "user_id_2": "category"})
        dataset = gjnn.dataset_preprocessing.prepare_out_of_sample_data(min_max_data_values, dataset)
        test = gjnn.dataloader.Dataset(dataset)
        loader = torch.utils.data.DataLoader(test, batch_size= batch_size, shuffle=False)

        total_acc = []
        total_loss = []
        print("IFIP_{} loaded.".format(j))
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
            if o % 50 == 0:
                print("Still working... {}/{}".format(o, len(loader)))

        loss = sum(total_loss) / len(total_loss)
        acc = sum(total_acc) / len(total_acc)
        writer.add_scalar('IFIP_' + str(j) + '/loss', loss, epoch + 1)
        writer.add_scalar('IFIP_' + str(j) + '/acc', acc, epoch + 1)
        print("Loss on out of sample IFIP_{} for epoch {} is: {:.4f}. Acc: {:.4f}\n".format(j, epoch + 1, loss, acc))




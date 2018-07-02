import torch.nn as nn
import torch

# At the moment each Siamese network branch has 3 hidden layers

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, siamese_layer_size, hidden_layer_size, output_size):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, siamese_layer_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(siamese_layer_size, siamese_layer_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(siamese_layer_size, siamese_layer_size)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(siamese_layer_size, siamese_layer_size)
        self.relu4 = nn.ReLU()


        self.fc5 = nn.Linear(siamese_layer_size * 2, hidden_layer_size)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Linear(hidden_layer_size, output_size)
        self.tanh_out = nn.Tanh()


    def forward_once(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.relu4(out)

        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Now we can merge the two branches of the siamese
        # and continue with a series of fully connected layers
        merged_out = torch.cat((output1, output2), 1)
        
        merged_branches = self.fc5(merged_out)
        merged_branches = self.relu5(merged_branches)

        merged_branches = self.fc6(merged_branches)
        merged_branches = self.relu6(merged_branches)

        merged_branches = self.fc7(merged_branches)
        merged_branches = self.tanh_out(merged_branches)

        return merged_branches

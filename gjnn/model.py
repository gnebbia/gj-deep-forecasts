import torch.nn as nn
import torch

# At the moment each Siamese network branch has 3 hidden layers

class SiameseNetwork(nn.Module):


    def __init__(self, input_size, siamese_layer_size, hidden_layer_size, output_size):
        super(SiameseNetwork, self).__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(1)

        def init_output_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.siamese_input = nn.Sequential(
            nn.Linear(input_size, siamese_layer_size),
            nn.ReLU()
        )
        self.siamese_input.apply(init_weights)

        self.siamese_fc1 = nn.Sequential(
            nn.Linear(siamese_layer_size,siamese_layer_size),
            nn.ReLU()
        )
        self.siamese_fc1.apply(init_weights)

        self.siamese_fc2 = nn.Sequential(
            nn.Linear(siamese_layer_size,siamese_layer_size),
            nn.ReLU()
        )
        self.siamese_fc2.apply(init_weights)

        self.siamese_fc3 = nn.Sequential(
            nn.Linear(siamese_layer_size,siamese_layer_size),
            nn.ReLU()
        )
        self.siamese_fc3.apply(init_weights)

        self.fc1 = nn.Sequential(
            nn.Linear(siamese_layer_size * 2, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU()
        )
        self.fc1.apply(init_weights)

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU()
        )
        self.fc2.apply(init_weights)

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU()
        )
        self.fc3.apply(init_weights)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_layer_size, 1),
            #nn.BatchNorm1d(output_size),
            #nn.Tanh()
        )
        self.output_layer.apply(init_output_weights)


    def forward_once(self, x):
        x = self.siamese_input(x)
        x = self.siamese_fc1(x)
        x = self.siamese_fc2(x)
        #x = self.siamese_fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Now we can merge the two branches of the siamese
        # and continue with a series of fully connected layers
        merged_out = torch.cat((output1, output2), 1)

        merged_out = self.fc1(merged_out)
        merged_out = self.fc2(merged_out)
        merged_out = self.fc3(merged_out)
        merged_out = self.output_layer(merged_out)

        return merged_out

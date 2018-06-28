import torch


class DistanceLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, output1, output2, tanh_output):
	# Computation of the euclidean distance between the forecasts
        e_dist = F.pairwise_distance(output1, output2) 
	
        # Assign variables to x and y just for more readability
        x = tanh_output
        y = e_dist
        loss_distance = ((((x-8)*172)**2 - ((y-8)*172)**2 + 2*((x)*172)*((y)*172)**2 + 1)/10**5) + 101.769

        return loss_distance


import torch
import numpy as np
import logging

dist_max = np.sqrt(2)
dist_min = -dist_max

logger = logging.getLogger(__name__)

class DistanceLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, dist_a, dist_b, tanh_output):
        #loss_distance = ( ( ( (x-8)*172)**2 - ( (y-8)*172 )**2 + 2*( (x)*172)*( (y)*172)**2 + 1)/10**5 ) + 101.769
        # Assign variables to x and y just for more readability
        x = tanh_output

        # Get the difference between distances from the outcome
        y = dist_b - dist_a

        # Normalize the difference between distances to be in the range [-1,1]
        #y = (2*(y - dist_min)/(dist_max - dist_min)) -1
        y = y.view(-1,1)

        # To not have all the operations on a single line
        # I preferred to keep track of InterMediate steps
        # through variables called 'im'
        x_squared = torch.pow(x, 2)
        y_squared = torch.pow(y, 2)
        xy = x * y
        x2y = torch.mul(xy, 2)

        im1 = torch.add(x_squared, -y_squared)
        im2 = torch.add(im1, -x2y)

        im3 = torch.mul(im2, 50)
        
        the_loss = torch.add(im3, 100)

        the_loss = torch.mean(the_loss)

        # Logging Loss Function computation
        logger.info("Size of x (tanh output) is: " + str(x))
        logger.info(x.size())
        logger.info("Size of y dist b - dist a is: " + str(y))
        logger.info(y.size())
        logger.info("Size of im1: x_squared - y_squared is : " + str(im1))
        logger.info(im1.size())
        logger.info("Size of im2: im1 - 2xy is : " + str(im2))
        logger.info(im2.size())
        logger.info("Size of im3: im2*50 is : " + str(im3))
        logger.info(im3.size())
        logger.info("Size of loss function is : " + str(the_loss))
        logger.info(the_loss.size())

        return the_loss

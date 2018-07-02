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
       logger.debug("Size of x (tanh output) is: " + str(x))
       logger.debug(x.size())
        logger.info("Size of y dist b - dist a is: " + str(y))
        logger.info(y.size())
        logger.info("Size of im1 x -8 is : " + str(im1))
        logger.info(im1.size())
        logger.info("Size of im2 im1 * 172 is : " + str(im2))
        logger.info(im2.size())
        logger.info("Size of im3 im2**2 is : " + str(im3))
        logger.info(im3.size())
        logger.info("Size of im4 y -8 is : " + str(im4))
        logger.info(im4.size())
        logger.info("Size of im5 im4*172 is : " + str(im5))
        logger.info(im5.size())
        logger.info("Size of im6 im5**2 is : " + str(len(im6)))
        logger.info(im6.size())
        logger.info("Size of im7 x*2 is : " + str(im7))
        logger.info(im7.size())
        logger.info("Size of im8 im7*172 is : " + str(im8))
        logger.info(im8.size())
        logger.info("Size of im9 im8*y is : " + str(im9))
        logger.info(im9.size())
        logger.info("Size of im10 im9**2 is : " + str(im10))
        logger.info(im10.size())
        logger.info("Size of im11 im3 - im6 is : " + str(im11))
        logger.info(im11.size())
        logger.info("Size of im12 im11 + im10 is : " + str(im12))
        logger.info(im12.size())
        logger.info("Size of im13 im12 + 1 is : " + str(im13))
        logger.info(im13.size())
        logger.info("Size of im14 im13 / 10**5 is : " + str(im14))
        logger.info(im14.size())
        logger.info("Size of final loss distance is : " + str(loss_distance))
        logger.info(loss_distance.size())

        # Get the difference between distances from the outcome
        y = dist_b - dist_a

        # Normalize the difference between distances to be in the range [-1,1]
        y = (2*(y - dist_min)/(dist_max - dist_min)) -1
        y = y.view(-1,1)

        # To not have all the operations on a single line
        # I preferred to keep track of InterMediate steps
        # through variables called 'im'
        im1 = torch.add(x, -8)
        im2 = torch.mul(im1, 172)
        im3 = torch.pow(im2, 2)

        im4 = torch.add(y, -8)
        im5 = torch.mul(im4, 172)
        im6 = torch.pow(im5, 2)
        
        im7 = torch.mul(x, 2)
        im8 = torch.mul(im7, 172)
        im9 = im8 * y

        im10 = torch.pow(im9, 2)

        im11 = torch.add(im3, -im6)

        im12 = torch.add(im11, im10)

        im13 = torch.add(im12, 1)

        im14 = torch.div(im13, 10**5)

        loss_distance = torch.add(im14, 101.769)

        loss_distance = loss_distance.mean()

        return loss_distance


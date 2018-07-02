import torch
import numpy as np

dist_max = np.sqrt(2)
dist_min = -dist_max

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
#        print("Size of x (tanh output) is: " + str(x))
#        print(x.size())

        y = dist_b - dist_a
        # Normalize the difference between distances to be in the range [-1,1]
        y = (2*(y - dist_min)/(dist_max - dist_min)) -1
        y = y.view(-1,1)
        print("Size of y dist b - dist a is: " + str(y))
        print(y.size())

        im1 = torch.add(x, -8)
#        print("Size of im1 x -8 is : " + str(im1))
#        print(im1.size())
        im2 = torch.mul(im1, 172)
#        print("Size of im2 im1 * 172 is : " + str(im2))
#        print(im2.size())
        im3 = torch.pow(im2, 2)
#        print("Size of im3 im2**2 is : " + str(im3))
#        print(im3.size())

        im4 = torch.add(y, -8)
#        print("Size of im4 y -8 is : " + str(im4))
#        print(im4.size())
        im5 = torch.mul(im4, 172)
#        print("Size of im5 im4*172 is : " + str(im5))
#        print(im5.size())
        im6 = torch.pow(im5, 2)
#        print("Size of im6 im5**2 is : " + str(len(im6)))
#        print(im6.size())
        
        im7 = torch.mul(x, 2)
#        print("Size of im7 x*2 is : " + str(im7))
#        print(im7.size())
        im8 = torch.mul(im7, 172)
#        print("Size of im8 im7*172 is : " + str(im8))
#        print(im8.size())
        im9 = im8 * y
#        print("Size of im9 im8*y is : " + str(im9))
#        print(im9.size())
        im10 = torch.pow(im9, 2)
#        print("Size of im10 im9**2 is : " + str(im10))
#        print(im10.size())

        im11 = torch.add(im3, -im6)
#        print("Size of im11 im3 - im6 is : " + str(im11))
#        print(im10.size())

        im12 = torch.add(im11, im10)
#        print("Size of im12 im11 + im10 is : " + str(im12))
#        print(im12.size())

        im13 = torch.add(im12, 1)
#        print("Size of im13 im12 + 1 is : " + str(im13))
#        print(im13.size())

        im14 = torch.div(im13, 10**5)
#        print("Size of im14 im13 / 10**5 is : " + str(im14))
#        print(im14.size())

        loss_distance = torch.add(im14, 101.769)
#        print("Size of final loss distance is : " + str(loss_distance))
#        print(loss_distance.size())

        loss_distance = loss_distance.sum()


        return loss_distance


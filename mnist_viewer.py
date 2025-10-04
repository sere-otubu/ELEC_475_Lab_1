
#########################################################################################################
#
#   ELEC 475 - Lab 1
#   Erhowvosere Otubu - 20293052
#   Mihran Asadullah - 20285090
#   Fall 2025
#

import torch
from torchvision.datasets import MNIST # Gets the MNIST dataset from torchvision
import torchvision.transforms as transforms # imports image transform utilities
import matplotlib.pyplot as plt # imports matplotlib for plotting


# downloads training partition to local directory ./data/mnist
train_transform = transforms.Compose([transforms.ToTensor()])

train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

# gets input from user for an index from 0 - 59999
idx = int(input("Enter an index between 0 and 59999: "))

# gets the image from trainset
plt.imshow(train_set.data[idx], cmap='gray')
# displays the image
plt.show()


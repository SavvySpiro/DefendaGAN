import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import numpy as np

from utils import Logger

def mnist_data(train=True):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5), (.5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=train, transform=compose, download=True)# Load data

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n


if __name__ == "__main__":
    data = mnist_data()# Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)
    for img, label in data_loader:
        plt.imshow(np.transpose(img[0], (1,2,0)))
        plt.title(label[0])
        plt.show()

    # fig, ax = plt.subplots(1)
    # ax.imshow(data[0][0], cmap='gray', interpolation='none')
    # ax.text(5, 1.5, f'predicted: {predicted.numpy()[0]}, actual: {label} --> {text}', bbox={'facecolor': 'white', 'pad': 10})
    # plt.show()
    # outputs = model(data)
    # predicted = torch.max(outputs.data, 1, keepdim=True)[1][0]
    # # print("label:     ", label)
    # # print("predicted: ", predicted)
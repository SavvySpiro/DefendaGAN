from torch import nn, optim
import torch
from torch.autograd.variable import Variable
from utils import vectors_to_images, images_to_vectors, Logger
from dataset import mnist_data
from matplotlib import pyplot as plt
import numpy as np

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 10
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 784
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    

def train_discriminator(optimizer, real_data, adv_data, all_labels):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(torch.cat((real_data, adv_data), dim=0))
    # Calculate error and backpropagate

    all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10)
    error_real = loss(prediction_real.float(), all_labels.float())
    error_real.backward()

    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real, prediction_real


def train_generator(optimizer, adv_batch, all_labels):
    N = adv_batch.size(0)    
    # Reset gradients
    optimizer.zero_grad()    
    # Try to classify adversarial data
    prediction = discriminator(adv_batch).float()
    # adversarial one hot encodings
    all_labels = torch.abs(torch.nn.functional.one_hot(all_labels, num_classes=10).float()-torch.ones((N,1), dtype=torch.float32))
    # Calculate error and backpropagate
    error = loss(prediction, all_labels.float())
    error.backward()    
    # Update weights with gradients
    optimizer.step()    
    # Return error
    return error


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

if __name__ == "__main__":    
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, weight_decay=0.01) # l2 regularization
    loss = nn.BCELoss()

    num_test_samples = 16
    test_noise = noise(num_test_samples)
    # Create logger instance
    
    data = mnist_data()# Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)

    logger = Logger(model_name='VGAN', data_name='MNIST')# Total number of epochs to train
    num_epochs = 60
    g_losses = []
    d_losses = []
    for epoch in range(num_epochs):
        for n_batch, (real_batch, labels) in enumerate(data_loader):
            N = real_batch.size(0)        
            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch[:(N+1)//2]))        
            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            unperturb_data = Variable(images_to_vectors(real_batch[(N+1)//2:]))   

            adv_data = generator(unperturb_data).detach()        
            # Train D
            d_error, d_pred = train_discriminator(d_optimizer, real_data, adv_data, labels)

            # 2. Train Generator        
            # # Generate fake data
            real_batch = Variable(images_to_vectors(real_batch))
            adv_batch = generator(real_batch)        
            # Train G
            g_error = train_generator(g_optimizer, adv_batch, labels)   

            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)        
            # Display Progress every few batches
            if (n_batch) % 100 == 0: 
                test_images = vectors_to_images(generator(real_batch))
                test_images = test_images.data            
                logger.log_images(
                    test_images, num_test_samples, 
                    epoch, n_batch, num_batches
                )
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred
                )

        d_losses.append(d_error.detach().numpy())
        g_losses.append(g_error.detach().numpy())
            
    


    fig = plt.figure()
    plt.plot(list(range(1, num_epochs+1)), d_losses, label="Discriminator")
    plt.plot(list(range(1, num_epochs+1)), g_losses, label="Generator")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

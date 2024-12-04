import comparison
from torch import nn, optim
import torch
from torch.autograd.variable import Variable
from utils import vectors_to_images, images_to_vectors, Logger
from dataset import mnist_data
from matplotlib import pyplot as plt
import numpy as np
import utils
import wandb
import config
from itertools import combinations
import copy

def gen_noise(size = config.Z_SHAPE):
    return torch.rand(size) * 2 - 1

def run():
    discriminator = Discriminator()
    generator = Generator()

    # Load data
    train_data = mnist_data(train=True)# Create loader with data, so that we can iterate over it
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_data = mnist_data(train=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Num batches
    num_batches = len(train_data_loader)

    # Log Stuff
    gen_losses = []
    dis_losses = []
    g_test_accuracy = []
    d_test_accuracy = []

    for epoch in range(config.NUM_EPOCHS):
        # train
        for n_batch, (real_batch, labels) in enumerate(train_data_loader):
            # Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))
            adv_data = generator(gen_noise())
            
            # Train Discriminator, freeze generator
            d_optimizer.zero_grad()
            d_loss = discriminator_loss(discriminator, real_data, adv_data)
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            adv_data = generator(gen_noise())
            g_loss = generator_loss(discriminator, adv_data)
            g_loss.backward()
            g_optimizer.step()

            # Add loss to list
            gen_losses.append(g_loss.data)
            dis_losses.append(d_loss.data)

            # test performance
            with torch.no_grad():
                test_real_data = Variable(images_to_vectors(test_data_loader.__iter__().next()[0]))
                test_adv_data = generator(gen_noise())
                d_test_accuracy.append(discriminator_loss(discriminator, test_real_data, test_adv_data))
                g_test_accuracy.append(generator_loss(discriminator, test_adv_data))

            
            



if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="DefendaGAN",
        config={
            "epochs": config.EPOCHS,
            "dataset":"MNIST"
        }
    )

    # initilize logger
    logger = Logger(model_name='VGAN', data_name='MNIST')# Total number of epochs to train

    # Create optimizers
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.DISCRIMINATOR_LR)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.GENERATOR_LR)

    # Create loss function
    CE_loss = nn.CrossEntropyLoss()
    MSE_loss = nn.MSELoss()
    
    
    # Train the GAN
    run()
from torch import nn, optim
import torch
from torch.autograd.variable import Variable
from utils import vectors_to_images, images_to_vectors, Logger
from dataset import mnist_data
from matplotlib import pyplot as plt
import numpy as np
import wandb

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 10
        
        '''
        {conv(32, 3, 3) + relu} * 2
        maxpool(2,2)
        {conv(64, 3, 3) + relu} * 2
        maxpool(2,2)
        FC(200) + relu
        FC(200) + relu
        fc(10) + softmax
        '''

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(200, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(self.fc(x))
            

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 784
        n_out = 784
        
        '''
        deconv1 (16,3,3) + RELU
        deconv2 (8,5,5) + RELU
        deconv3 (1,2,2) + tanh
        '''

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1),
            nn.ReLU() 
        )

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(5,5), stride=1),
            nn.ReLU()
        )

        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(2,2), stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        return x.view(x.size(0), 784)


def train_discriminator(d_optimizer, real_data, adv_data, all_labels):
    N = real_data.size(0)
    # Reset gradients
    d_optimizer.zero_grad()
    # Freeze generator
    generator.eval()
    # Generate adversarial data
    adv_data = torch.clamp(generator(real_data).detach(), -1,1) + real_data
    adv_data = Variable(images_to_vectors(adv_data))
    real_data = Variable(images_to_vectors(real_data))
    
    # Get discriminator outputs for real and fake data
    prediction_real = discriminator(torch.cat((real_data, adv_data), dim=0))

    # first half is true data, second half is perturbed. 
    all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10)
    error_real = CEloss(prediction_real, torch.cat((all_labels, all_labels), dim=0))
    error_real.backward()

    # 1.3 Update weights with gradients
    d_optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real, prediction_real


def train_generator(optimizer, real_batch, all_labels, alpha=0.001):
    N = adv_batch.size(0)    
    # Reset gradients
    optimizer.zero_grad()    
    # freeze discriminator
    discriminator.eval()
    # Generate fake data
    adv_batch = torch.clamp(generator(real_batch) + real_batch, -1,1)        
    adv_batch = Variable(images_to_vectors(adv_batch))
    real_batch = Variable(images_to_vectors(real_batch))

    # Try to classify adversarial data 
    prediction = (discriminator(adv_batch)).float()

    # create labels where true label is 0 and remaining are equally distributed
    all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10)
    all_labels[all_labels == 0] = 1.0 / (all_labels.shape[1] - 1)
    all_labels[all_labels == 1] = 0

    # Error is 
    error = MSEloss(prediction.float(), all_labels.float()) + torch.sum(torch.abs(adv_batch + 1)) * alpha

    # print("error", error)
    # Add L1 regularization
    error += torch.sum(torch.abs(adv_batch + 1)) * 0.0001

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

def normalize_loss(ema, val, alpha=0.5):
    return alpha * val + (1 - alpha) * ema

if __name__ == "__main__":    

    wandb.init(
        project="DefendaGAN",
        config={
            "epochs":60,
            "dataset":"MNIST"
        }
    )
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    initial_weight_decay = 0.001
    final_weight_decay = 0.00001

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002) #, weight_decay=initial_weight_decay) # l2 regularization
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()

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
    g_EMA = -1
    d_EMA = -1
    for epoch in range(num_epochs):
        for n_batch, (real_batch, labels) in enumerate(data_loader):
            N = real_batch.size(0)        
            # 1. Train Discriminator
            real_data = real_batch
            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            unperturb_data = torch.clone(real_data).detach()

            # discriminator winning
            if len(g_losses) != 0 and g_losses[-1] > d_losses[-1]:

                # 1. Train Generator        
                g_error = train_generator(d_optimizer, real_batch, labels)   

                #2.  Train D

                d_error, d_pred = train_discriminator(g_optimizer, real_data, adv_data, labels)

            # Generator winning or no data yet
            elif len(g_losses) == 0 or g_losses[-1] <= d_losses[-1]:
                adv_data = torch.clamp(generator(unperturb_data).detach(), -1,1) + unperturb_data      
                # 1. Train Discriminator
                adv_data = Variable(images_to_vectors(adv_data))
                real_data = Variable(images_to_vectors(real_data))
                d_error, d_pred = train_discriminator(d_optimizer, real_data, adv_data, labels)

                # 2. Train Generator        
                # # Generate fake data
                real_batch = real_batch
                adv_batch = torch.clamp(generator(real_batch) + real_batch, -1,1)
                # print(generator(real_batch)[0])
                # Train G
                adv_batch = Variable(images_to_vectors(adv_batch))
                g_error = train_generator(g_optimizer, adv_batch, labels)   

            # seen range of gen outputs
            # print(torch.max(generator(real_batch)), torch.min(generator(real_batch)))

            # print("raw error", d_error, g_error)

        
            # normalize data for comparison
            # if g_EMA == -1:
            #     g_EMA = g_error
            #     d_EMA = d_error
            # else:
            #     g_ema = normalize_loss(g_EMA, g_error)
            #     g_error = g_error / g_EMA

            #     d_ema = normalize_loss(d_EMA, d_error)
            #     d_error = d_error / d_EMA 

            # normalize data for comparison via sigmoid
            # g_error = torch.sigmoid(g_error)
            # d_error = torch.sigmoid(d_error)

            # print("normalized error", d_error, g_error)

            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)        

            # wandb.log({"Error": wandb.plot.line_series(xs=list(range(1, num_epochs+1)), 
            #                                             ys=[d_losses, g_losses], 
            #                                             keys=["Discriminator", "Generator"],
            #                                             title="Losses",
            #                                             xname="Epoch",)})

            wandb.log({"Generator Error": g_error, "Discriminator Error": d_error})

            # Display Progress every few batches
            if (n_batch) % 100 == 0: 
                # plt.imshow(torch.transpose(torch.reshape(real_batch, (-1, 28, 28))[1:2, :, :], 0, 2))
                # plt.show()

                test_images = vectors_to_images(torch.clamp(generator(real_batch) + real_batch, -1,1))
                test_images = test_images.data   

                # plt.imshow(torch.transpose(torch.reshape(test_images, (-1, 28, 28))[1:2, :, :], 0, 2))
                # plt.show()
                
                logger.log_images(
                    test_images, num_test_samples, 
                    epoch, n_batch, num_batches
                )
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred
                )

                # Log discriminator error on perturbed and unperturbed data
                real_batch_vec = Variable(images_to_vectors(real_batch))
                unperturbed = discriminator(real_batch_vec)
                perturbed = discriminator(Variable(images_to_vectors(torch.clamp(generator(real_batch) + real_batch, -1,1))))

                # print(unperturb_data.shape, perturbed.shape, real_batch_vec.shape)
                unperturbed_accuracy = torch.sum(torch.argmax(unperturbed, dim=1) == labels).float() / len(labels)
                perturbed_accuracy = torch.sum(torch.argmax(perturbed, dim=1) == labels).float() / len(labels)

                wandb.log({"Unperturbed Accuracy": unperturbed_accuracy, "Perturbed Accuracy": perturbed_accuracy})

        d_losses.append(d_error.detach().numpy())
        g_losses.append(g_error.detach().numpy())

        # update weight decay
        g_optimizer.param_groups[0]['weight_decay'] = initial_weight_decay + (final_weight_decay - initial_weight_decay) * (epoch / num_epochs)

    
    fig = plt.figure()
    plt.plot(list(range(1, num_epochs+1)), d_losses, label="Discriminator")
    plt.plot(list(range(1, num_epochs+1)), g_losses, label="Generator")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    wandb.finish()


'''
Notes for Alenna's Sanity:
1. see why the test set is all 0s, what
2. display change in perturbations to the same image/set of 0-9 over epochs
3. display accuracy on unperturbed and perturbed sets over epochs
4. display examples of correct and incorrect classifications over epochs
5. add maturation schedule to regularization of generator
6. Plot performance against fgsm over time 
7. 
'''
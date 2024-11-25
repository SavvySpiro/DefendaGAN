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

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        
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
            nn.Dropout(0.5),    
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.out = nn.Sequential(
            nn.Linear(200, 10),
            nn.Softmax(dim=1)
        )

        
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)



    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        x = self.out(x)
        # print(x.shape)
        return x
    
    def train_d(self, generator, real_data, adv_batch, all_labels):
        N = real_data.size(0)
        # Reset gradients
        self.optimizer.zero_grad()
        # detach generator output
        adv_batch = [b.detach() for b in adv_batch]
        # detatch generator
        for param in generator.parameters():
            param.requires_grad = False

        # Get discriminator outputs for real and fake data
        prediction_real = self(real_data)
        prediction_fake = [self(b) for b in adv_batch]


        # first half is true data, second half is perturbed. 
        all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10)
        # ensure inputs to CE are float
        prediction_real = prediction_real.float()
        prediction_fake = [p.float() for p in prediction_fake]
        all_labels = all_labels.float()

        # choose prediction fake which causes the worst loss
        prediction_fake = min(prediction_fake, key=lambda x: CEloss(x, all_labels))

        error = CEloss(prediction_real, all_labels) + CEloss(prediction_fake, all_labels)
        error.backward()

        # 1.3 Update weights with gradients
        self.optimizer.step()

        # reattach generator
        for param in generator.parameters():
            param.requires_grad = True
        
        # Return error and predictions for real and fake inputs
        return error, torch.cat((prediction_real, prediction_fake), 0)
            

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        
        '''
        deconv1 (16,3,3) + RELU
        deconv2 (8,5,5) + RELU
        deconv3 (1,2,2) + tanh
        '''

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=1),
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

        self.Dropout = nn.Dropout(0.5)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)


    def forward(self, x):
        # x = x.view(x.size(0), 1, 28, 28)
        # print(x.shape)
        x = self.dconv1(x)
        x = self.Dropout(x)
        # print(x.shape)
        x = self.dconv2(x)
        x = self.Dropout(x)
        # print(x.shape)
        x = self.dconv3(x)
        # print(x.shape)
        return x.view(-1, 1, 28, 28)
    
    def train_g(self, discriminator, real_batch, adv_batch, all_labels, alpha=0.0001):
        N = adv_batch.size(0)    
        # Reset gradients
        self.optimizer.zero_grad()    
        # freeze discriminator
        for param in discriminator.parameters():
            param.requires_grad = False

        # Try to classify adversarial data 
        prediction = (discriminator(adv_batch)).float()

        if config.GEN_LOSS == "even":
            # create labels where true label is 0 and remaining are equally distributed
            all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10).float()
            # print(all_labels.shape)
            all_labels[all_labels == 0] = 1.0 / (all_labels.shape[1] - 1)
            # print(all_labels)
            all_labels[all_labels == 1] = 0
        if config.GEN_LOSS == "target":
            # choose random target that isnt the true label
            all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10).float()
            zero_positions = (all_labels == 0)
            random_choices = torch.multinomial(zero_positions.float(), 1).squeeze(1)  # (batch_size,)
            all_labels_targeted = torch.nn.functional.one_hot(random_choices, num_classes=10).float()
            # enasure true label is 0
            assert(all_labels_targeted[all_labels == 1].sum() == 0)

            all_labels = all_labels_targeted
        if config.GEN_LOSS == "easiest":
            # choosing the target that is second highest
            targets = torch.argsort(all_labels, dim=1, descending=True)
            all_labels = torch.nn.functional.one_hot(targets[:, 1], num_classes=10).float()

        # ensure inputs to CE are float
        prediction = prediction.float()
        all_labels = all_labels.float()

        # Error is 
        error = MSEloss(prediction, all_labels) 
        
        if config.REGULARIZATION:
            error += alpha * torch.sum(torch.abs(real_batch - adv_batch))
        # print("error", error)

        error.backward()    
        # Update weights with gradients
        self.optimizer.step()    
        # Return error
        # reattach discriminator
        for param in discriminator.parameters():
            param.requires_grad = True

        return error

    
class GeneratorNetGroup(torch.nn.Module):
    '''
    Combines training of several generators
    '''
    def __init__(self, num_generators):
        super(GeneratorNetGroup, self).__init__()
        self.generators = [GeneratorNet() for n in range(num_generators)]

    def forward(self, x):
        vals = [gen(x) for gen in self.generators]
        return vals
    
    def backward(self, x):
        for gen in self.generators:
            gen.backward(x)
    
    # override .parameters() to return all parameters
    def parameters(self):
        return [param for gen in self.generators for param in gen.parameters()]
    
    def train_g(self, discriminator, real_batch, adv_batch, all_labels, kl_weight=0.1): #higher kl_weight means more diversity
        # Train each generator while maximizing kl divergence between all generators
        noise_g = noise(config.Z_SHAPE, real_batch.size(0))
        outputs = [gen(noise_g) for gen in self.generators]

        if config.N_GENERATOR == 1:
            return [self.generators[0].train_g(discriminator, real_batch, adv_batch[0], all_labels)]
        
        diversity_loss = 0
        pairs = list(combinations(outputs, 2))
        for pair in pairs:
            i = pair[0]
            j = pair[1]
            # prepare for kl divergence
            i[i == 0] = 1e-6
            j[j == 0] = 1e-6
            # scale to 0-1
            i = (i - i.min()) / (i.max() - i.min())
            j = (j - j.min()) / (j.max() - j.min())
            # normalize
            i = i / i.sum()
            j = j / j.sum()

            kl_ij = torch.nn.functional.kl_div(i, j, reduction='batchmean')
            kl_ji = torch.nn.functional.kl_div(j, i, reduction='batchmean')
            # print(kl_ij, kl_ji)
            diversity_loss += kl_ij + kl_ji

        if pairs:
            diversity_loss /= len(pairs)

        # Train each generator
        errors = []
        for i, gen in enumerate(self.generators):
            gen_loss = gen.train_g(discriminator, real_batch, adv_batch[i], all_labels)
            errors.append(gen_loss - kl_weight * diversity_loss)

        # choose the generator with the lowest loss
        return min(errors)

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

def noise(size, batch_size=config.BATCH_SIZE):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.rand(batch_size, *size) * 2 - 1)
    return n


def normalize_loss(ema, val, alpha=0.5):
    return alpha * val + (1 - alpha) * ema



def run():
    # Initialize discriminator and generator(s)
    discriminator = DiscriminatorNet()
    generator = GeneratorNetGroup(config.N_GENERATOR)


    # Load data
    data = mnist_data()# Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    # Num batches
    num_batches = len(data_loader)

    num_epochs = 50
    g_losses = []
    d_losses = []
    g_test_acc = None
    d_test_acc = None
    for epoch in range(num_epochs):
        for n_batch, (real_batch, labels) in enumerate(data_loader):
            N = real_batch.size(0)       

            noise_data = noise(config.Z_SHAPE, real_batch.size(0))
            adv_batch = [torch.clamp(g + real_batch, -1, 1) for g in generator(noise_data)]
            # discriminator winning
            if g_test_acc != None and g_test_acc > d_test_acc and config.TRAINING_ALTERNATE:

                #1. Train Generator
                g_error = generator.train_g(discriminator, real_batch, adv_batch, labels)   

                #2.  Train Discriminator
                d_error, d_pred = discriminator.train_d(generator, real_batch, adv_batch, labels)

            # Generator winning or no data yet
            elif d_test_acc == None or g_test_acc <= d_test_acc:

                # 1. Train Discriminator
                d_error, d_pred = discriminator.train_d(generator, real_batch, adv_batch, labels)

                # 2. Train Generator        
                g_error = generator.train_g(discriminator, real_batch, adv_batch, labels)   
        
            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)        

            wandb.log({"Generator Error": g_error, "Discriminator Error": d_error})

            # Display Progress every few batches
            if (n_batch) % 100 == 0:
                # Sample some noise and generate test images from the current batch
                test_images_noise = generator(noise(config.Z_SHAPE))
                test_images = [torch.clamp(g + real_batch, -1, 1) for g in test_images_noise]
                
                # Log Noise in wandb (default)
                for i, img in enumerate(test_images_noise):
                    logger.log_images("Generated Noise {}".format(i), img, 100, epoch, n_batch, num_batches)
                # Log Generated Images in wandb
                for i, img in enumerate(test_images):
                    logger.log_images("Generated {}".format(i), img, 100, epoch, n_batch, num_batches)

                # Display status Logs
                logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred)

                # Log discriminator error on perturbed and unperturbed data       
                discriminator.eval()    
                unperturbed = discriminator(Variable(images_to_vectors(real_batch)))
                perturbed = [discriminator(Variable(images_to_vectors(p))) for p in test_images]

                # calculate accuracy
                unperturbed_accuracy = torch.sum(torch.argmax(unperturbed, dim=1) == labels).float() / len(labels)
                perturbed_accuracy = [torch.sum(torch.argmax(p, dim=1) == labels).float() / len(labels) for p in perturbed]

                g_test_acc = min(perturbed_accuracy)
                d_test_acc = unperturbed_accuracy

                wandb.log({"Unperturbed Accuracy": unperturbed_accuracy, "Min Perturbed Accuracy": min(perturbed_accuracy), "Max Perturbed Accuracy": max(perturbed_accuracy)})

                # Record fgsm error compared to perturbed data
                fgsm_perturbed_pred, _ = comparison.apply_fgsm_attack(discriminator, 0.1, real_batch, labels, CEloss)
                fgsm_perturbed_accuracy = torch.sum(torch.argmax(fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
                wandb.log({"FGSM Perturbed Accuracy": fgsm_perturbed_accuracy})


        d_losses.append(d_error.detach().numpy())
        g_losses.append(g_error.detach().numpy())

    
    fig = plt.figure()
    plt.plot(list(range(1, num_epochs+1)), d_losses, label="Discriminator")
    plt.plot(list(range(1, num_epochs+1)), g_losses, label="Generator")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    wandb.finish()

    # Save models
    utils.save_model(generator, "generator", config)
    utils.save_model(discriminator, "discriminator", config)

if __name__ == "__main__":    

    # Initialize wandb
    wandb.init(
        project="DefendaGAN",
        config={
            "epochs":50,
            "dataset":"MNIST"
        }
    )

    # Initialize logger
    logger = Logger(model_name='VGAN', data_name='MNIST')# Total number of epochs to train

    # Initialize Loss functions
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()
    #to run experiments, change configs here before running
    run()
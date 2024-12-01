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

class mnist_classifier(nn.Module):
  def __init__(self, mnist_dim, layer_size, classes):
    super(mnist_classifier, self).__init__()
    self.flatten = nn.Flatten()
    self.input_layer = nn.Linear(mnist_dim*mnist_dim, mnist_dim)
    self.activation_layer = nn.ReLU()
    self.output_layer = nn.Linear(mnist_dim, classes)
    
  def forward(self, input):
    input = self.flatten(input)
    input = self.input_layer(input)
    input = self.activation_layer(input)
    input = self.output_layer(input)
    return input
  
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

        # choose prediction fake (produced by the n generators) which causes the worst loss
        prediction_fake = max(prediction_fake, key=lambda x: CEloss(x, all_labels))

        if config.DISABLE_ATTACK:
            error = CEloss(prediction_real, all_labels)
        else:
            error = CEloss(prediction_real, all_labels) + CEloss(prediction_fake, all_labels)
            
        error.backward()

        # 1.3 Update weights with gradients
        self.optimizer.step()

        # reattach generator
        for param in generator.parameters():
            param.requires_grad = True
        
        # Return error and predictions for real and fake inputs
        error = error.detach()
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
    
    def train_g(self, discriminator, real_batch, adv_batch, all_labels, diversity_loss=0, alpha=0.0001):
        N = adv_batch.size(0)    
        # Reset gradients
        self.optimizer.zero_grad()    

        if diversity_loss != 0:
            diversity_loss = diversity_loss.detach()
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
            # ensure true label is 0
            assert(all_labels_targeted[all_labels == 1].sum() == 0)

            all_labels = all_labels_targeted
        if config.GEN_LOSS == "easiest":
            # choose the target predictions that is second highest, likely the easiest to fool
            targets = torch.argsort(prediction, dim=1, descending=True)
            all_labels = torch.nn.functional.one_hot(targets[:, 1], num_classes=10).float()
            all_labels = all_labels
        if config.GEN_LOSS == "easiest_exp":
            targets = torch.argsort(prediction, dim=1, descending=True)
            # Create one-hot encoded labels
            all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10).float()

            # Set the true label to 0 and others to decreasing values
            for i in range(all_labels.shape[0]):
                # Get the index of the true label (assuming target is the first element in the sorted targets)
                true_label = (all_labels[i] == 1).nonzero(as_tuple=True)[0].item()  # Extract the index of the true label
                predicted_label = (targets[i] == 0).nonzero(as_tuple=True)[0].item()

                # only if the prediction is correct, try to fool the discrimination by assigning 0 probably to the 
                # actual class and exponentially decreasing probability from the second most likely class onwards
                # MODIFICATION to assigning uniform probablity as in the "even" option
                if true_label == predicted_label: 
                    # print("prediction: ", prediction[i])
                    # print("target: ", targets[i])
                    # print("sorted: ", prediction[i][targets[i]])

                    # Generate exponentially decreasing values
                    base = 0.5  # You can choose the base for exponential decay
                    values = base ** torch.arange(9).float()
                    # Normalize to make it a probability distribution
                    values = values / values.sum()
                    values = torch.cat((torch.tensor([0]), values))

                    # # Assign the decreasing values to the remaining labels
                    # print("BEFORE: ", all_labels[i])
                    # print(f"prediction[i] :     {prediction[i]}")
                    # print(f"targets[i] :        {targets[i] }")
                    # print(f"values[i] :         {values}")
                    # print(" *****************************************")
                    for ind in range(10):
                        # print("NEW probability: ", values[ind])
                        destination = torch.nonzero(targets[i] == ind, as_tuple=True)[0][0]
                        # print("index of where it's going ", destination)
                        all_labels[i][destination] = values[ind]
                        # print("UPDATE: ", all_labels[i])
                        # print(" *****************************************")

                else: # just assign uniform probably to all other classes, assign true class to be zero
                    all_labels[i][all_labels[i] == 0] = 1.0 / (all_labels[i].shape[0] - 1)
                    # print(all_labels)
                    all_labels[i][all_labels[i] == 1] = 0.0

        # ensure inputs to CE are float
        prediction = prediction.float()
        all_labels = all_labels.float()

        # Error is 
        error = MSEloss(prediction, all_labels)
        
        if config.REGULARIZATION:
            # enforce similarty of perturbed and unperturbed images
            error += alpha * torch.sum(torch.abs(real_batch - adv_batch))
        # print("error", error)

        error.backward()    
        # Update weights with gradients
        self.optimizer.step()    
        # Return error
        # reattach discriminator
        for param in discriminator.parameters():
            param.requires_grad = True

        error = error.detach()
        return error + diversity_loss, error

    
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
    
    def train_g(self, discriminator, real_batch, adv_batch, all_labels, kl_weight=config.KL_WEIGHT): #higher kl_weight means more diversity
        # Train each generator while maximizing kl divergence between all generators (for diversity amoung generators)
        noise_g = noise(config.Z_SHAPE, real_batch.size(0))
        outputs = [gen(noise_g) for gen in self.generators] # get an example of the random data outputs

        if config.N_GENERATOR == 1:
            return self.generators[0].train_g(discriminator, real_batch, adv_batch[0], all_labels)
        
        diversity_loss = {}
        # pairs = list(combinations(outputs, 2)) # wrong
        pairs = list(combinations(enumerate(outputs), 2))  # Pair outputs with generator indices

        # calculate the kl divergence between each pair of generators
        # for pair in pairs:
        for (idx_i, i), (idx_j, j) in pairs:
            # i = pair[0]
            # j = pair[1]
            # prepare for kl divergence, maintain properties necessary for kl divergence
            # non-zero postive numbers
            # i[i == 0] = 1e-6
            # j[j == 0] = 1e-6
            # i = i.clamp(min=1e-6)  # Ensure no zero values
            # j = j.clamp(min=1e-6)
            # scale to 0-1
            # i = (i - i.min()) / (i.max() - i.min())
            # j = (j - j.min()) / (j.max() - j.min())
            
            i = ((i+1) / 2) + 1e-12
            j = ((j+1) / 2) + 1e-12
            # normalize
            i = i / i.sum()
            j = j / j.sum()

            kl_ij = torch.nn.functional.kl_div(i.log(), j, reduction='sum')
            kl_ji = torch.nn.functional.kl_div(j.log(), i, reduction='sum')

            # unlogd
            # kl_ij = torch.exp(kl_ij)
            # kl_ji = torch.exp(kl_ji)
            
            if torch.isnan(kl_ij) or torch.isnan(kl_ji) or torch.isinf(kl_ij) or torch.isinf(kl_ji):
                print(f"NaN/Inf encountered! kl_ij: {kl_ij}, kl_ji: {kl_ji}, i: {i}, j: {j}")

            # total diversity value
            # diversity_loss[(i, j)] =  kl_ij + kl_ji #wrong
            diversity_loss[(idx_i, idx_j)] = kl_ij + kl_ji
            # print("KL :", kl_ij, kl_ji)

            # kl div = how different are the output distributions of this pair of generators

        # Train each generator
        errors = [] 
        if config.N_GENERATOR == 1:
            return self.generators[0].train_g(discriminator, real_batch, adv_batch[0], all_labels)
        
        for i, gen in enumerate(self.generators):
            # consider the generator's individual performance WITH how seperate it's performance is relative to other generators
            # grab the kl divergence for the current generator being referenced
            # diversity_loss_curr = torch.tensor(np.sum([value for key,value in diversity_loss.items() if gen in key]))

            diversity_loss_curr = torch.scalar_tensor(0.0)
            for key, value in diversity_loss.items():
                if i in key:
                    diversity_loss_curr += value
            
            gen_loss, just_gen_loss = gen.train_g(discriminator, real_batch, adv_batch[i], all_labels, kl_weight*diversity_loss_curr)

            errors.append(gen_loss)
            wandb.log({f"Gen. {i} Diversity Loss": diversity_loss_curr})
            wandb.log({f"Gen. {i} Generator Loss": just_gen_loss})
        
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

    num_epochs = config.EPOCHS
    g_losses = []
    d_losses = []
    g_test_acc = None
    d_test_acc = None
    for epoch in range(num_epochs):
        for n_batch, (real_batch, labels) in enumerate(data_loader):
            N = real_batch.size(0)       

            noise_data = noise(config.Z_SHAPE, real_batch.size(0))
            if config.DISABLE_ATTACK:
                adv_batch = [torch.clamp(g*0 + real_batch, -1, 1) for g in generator(noise_data)]
            else:
                adv_batch = [torch.clamp(g + real_batch, -1, 1) for g in generator(noise_data)]
            # discriminator winning
            if g_test_acc != None and g_test_acc > d_test_acc and config.TRAINING_ALTERNATE:

                #1. Train Generator
                g_error, g_diversity = generator.train_g(discriminator, real_batch, adv_batch, labels)   

                #2.  Train Discriminator
                d_error, d_pred = discriminator.train_d(generator, real_batch, adv_batch, labels)

            # Generator winning or no data yet
            elif d_test_acc == None or g_test_acc <= d_test_acc:

                # 1. Train Discriminator
                d_error, d_pred = discriminator.train_d(generator, real_batch, adv_batch, labels)

                # 2. Train Generator        
                g_error, g_diversity = generator.train_g(discriminator, real_batch, adv_batch, labels)   
        
            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)        

            wandb.log({"Generator Error": g_error, "Discriminator Error": d_error})

            # Display Progress every few batches
            if (n_batch) % config.BATCH_SIZE == 0:
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

                # Record pgd error compared to perturbed data
                epsilon = 0.1
                alpha = 0.005
                n_iters = 100
                pgd_perturbed_pred, _ = comparison.apply_pgd_attack(discriminator, epsilon, alpha, n_iters, real_batch, labels, CEloss)
                pgd_perturbed_accuracy = torch.sum(torch.argmax(pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
                wandb.log({"PGD Perturbed Accuracy": pgd_perturbed_accuracy})


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
    utils.save_model(generator, config, name="generator_")
    utils.save_model(discriminator, config, name="discriminator_")

if __name__ == "__main__":    

    # Initialize wandb
    wandb.init(
        project="DefendaGAN",
        config={
            "epochs": config.EPOCHS,
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
import comparison
from torch import nn, optim
import torch
from torch.autograd.variable import Variable
from utils import vectors_to_images, images_to_vectors, Logger
from dataset import mnist_data, cifar10_data
from matplotlib import pyplot as plt
import numpy as np
import utils
import wandb
import config
from itertools import combinations
import copy

class Mnist_Classifier(nn.Module):
    def __init__(self, mnist_dim, classes):
        super(Mnist_Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(mnist_dim*mnist_dim, mnist_dim)
        self.activation_layer = nn.ReLU()
        self.output_layer = nn.Linear(mnist_dim, classes)
        self.optimizer = optim.Adam(self.parameters(), lr=config.DISCRIMINATOR_LR)
        
    def forward(self, input):
        input = self.flatten(input)
        input = self.input_layer(input)
        input = self.activation_layer(input)
        input = self.output_layer(input)
        return input
    
    def train_s(self, input, target):
        self.optimizer.zero_grad()
        output = self(input).float()
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        loss = nn.CrossEntropyLoss()
        loss = loss(output, target)
        loss.backward()
        self.optimizer.step()
        return loss
  
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, img_in_channels=1):
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
            nn.Conv2d(in_channels=img_in_channels, out_channels=32, kernel_size=(3,3), stride=1),
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
            nn.Linear(1600, 200),
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

        
        self.optimizer = optim.Adam(self.parameters(), lr=config.DISCRIMINATOR_LR)

    def forward(self, x):
        x = x.view(x.size(0), config.Z_SHAPE[0], config.Z_SHAPE[1], config.Z_SHAPE[2])
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

        if any([torch.isnan(b).any() or torch.isinf(b).any() for b in adv_batch]):
            raise ValueError("Adversarial batch contains nan or inf values")
        
        # Reset gradients
        self.optimizer.zero_grad()
        # detach generator output
        adv_batch = [b.detach() for b in adv_batch]
        # detatch generator
        for param in generator.parameters():
            param.requires_grad = False

        # for testing: black out all adv batch and see if the discriminator can still predict the correct label
        # adv_batch = [torch.zeros_like(b) for b in adv_batch]

        # Get discriminator outputs for real and fake data
        prediction_real = self(real_data)
        prediction_fake = [self(b) for b in adv_batch]

        # first half is true data, second half is perturbed. 
        all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10).float()
        # use softmax to get the probability of the true label
        all_labels[all_labels == 1] = 0.9
        all_labels[all_labels == 0] = 0.1

        # ensure inputs to CE are float
        prediction_real = prediction_real.float()
        prediction_fake = [p.float() for p in prediction_fake]

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

        if config.DISABLE_ATTACK:
            return error, prediction_real, (CEloss(prediction_real, all_labels), torch.tensor(0.0))
        
        return error, torch.cat((prediction_real, prediction_fake), 0), (CEloss(prediction_real, all_labels), CEloss(prediction_fake, all_labels))
            

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, img_in_channels=1):
        super(GeneratorNet, self).__init__()
        
        '''
        deconv1 (16,3,3) + RELU
        deconv2 (8,5,5) + RELU
        deconv3 (1,2,2) + tanh
        '''

        self.input_transform = nn.Sequential(
            nn.Conv2d(in_channels=img_in_channels, out_channels=16, kernel_size=8, stride=1),  # From (1x28x28) to (16x21x21)
            nn.ReLU()
        )

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=1), # From (16x21x21) to (16x23x23)
            nn.ReLU() 
        )

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(5,5), stride=1), # From (16x23x23) to (8x27x27)
            nn.ReLU()
        )

        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=img_in_channels, kernel_size=(2,2), stride=1), # From (8x27x27) to (1x28x28)
            nn.Tanh()
        )

        self.Dropout = nn.Dropout(0.5)

        self.optimizer = optim.Adam(self.parameters(), lr=config.GENERATOR_LR)


    def forward(self, x):
        # x = x.view(x.size(0), 1, 28, 28)
        # print(x.shape)
        x = self.input_transform(x)
        x = self.dconv1(x)
        x = self.Dropout(x)
        # print(x.shape)
        x = self.dconv2(x)
        x = self.Dropout(x)
        # print(x.shape)
        x = self.dconv3(x)
        # print(x.shape)
        return x.view(-1, config.Z_SHAPE[0], config.Z_SHAPE[1], config.Z_SHAPE[2])
    
    def train_g(self, discriminator, real_batch, adv_batch, all_labels, diversity_loss=0, alpha=config.REG_STRENGTH):
        if config.DISABLE_ATTACK:
            return 0, 0
        

        if any([torch.isnan(b).any() or torch.isinf(b).any() for b in adv_batch]):
            raise ValueError("Adversarial batch contains nan or inf values")
        
        # Reset gradients
        self.optimizer.zero_grad()    

        # freeze discriminator
        for param in discriminator.parameters():
            param.requires_grad = False

        if config.GEN_LOSS == "even":
            # create labels where true label is 0 and remaining are equally distributed
            all_labels = torch.nn.functional.one_hot(all_labels, num_classes=10).float()
            # print(all_labels.shape)
            all_labels[all_labels == 0] = 1.0 / (all_labels.shape[1] - 1)
            # print(all_labels)
            all_labels[all_labels == 1] = 0.0
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
        prediction = (discriminator(adv_batch)).float()
        all_labels = all_labels.float()

        # Error is 
        error = MSEloss(prediction, all_labels)
        # print("MSE Loss: ", error)
        
        if config.REGULARIZATION:
            # enforce similarty of perturbed and unperturbed images L2 NORMALIZATION
            error += alpha * torch.sum(torch.linalg.norm(real_batch - adv_batch, ord=2, dim=1))
            # print("reg loss: ", torch.sum(torch.abs(real_batch - adv_batch)))
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
    def __init__(self, num_generators, img_in_channels=1):
        super(GeneratorNetGroup, self).__init__()
        self.generators = [GeneratorNet(img_in_channels) for n in range(num_generators)]

    def forward(self, x):
        # add noise to x
        x = x + noise(config.Z_SHAPE, x.size(0))
        vals = [gen(x) for gen in self.generators]
        return vals
    
    def backward(self, x):
        for gen in self.generators:
            gen.backward(x)
    
    # override .parameters() to return all parameters
    def parameters(self):
        return [param for gen in self.generators for param in gen.parameters()]
    
    def state_dict(self):
        return [gen.state_dict() for gen in self.generators]
    
    def train_g(self, discriminator, real_batch, adv_batch, all_labels, kl_weight=config.KL_WEIGHT): #higher kl_weight means more diversity

        # ensure adv batch is not nan or inf
        if any([torch.isnan(b).any() or torch.isinf(b).any() for b in adv_batch]):
            raise ValueError("Adversarial batch contains nan or inf values")
        # if only one generator, return the loss for that generator
        if config.N_GENERATOR == 1:
            return self.generators[0].train_g(discriminator, real_batch, adv_batch[0], all_labels)
        
        # Train each generator while maximizing kl divergence between all generators (for diversity amoung generators)
        # noise_g = noise(config.Z_SHAPE, real_batch.size(0))
        outputs = [gen(real_batch) for gen in self.generators] # get an example of the random data outputs
        # check if outputs are the exact same

        if config.DIVERSITY_TYPE == "KL":
            stacked_outputs = torch.stack(outputs, dim=0)
            flattened = stacked_outputs.view(config.N_GENERATOR, config.BATCH_SIZE, -1)
            softmax_outputs = torch.nn.functional.softmax(flattened, dim=2)
            log_softmax_outputs = torch.log(softmax_outputs + 1e-12)

            kl_ij = (softmax_outputs.unsqueeze(0) * (log_softmax_outputs.unsqueeze(0) - log_softmax_outputs.unsqueeze(1))).sum(dim=3)

            kl_ji = (softmax_outputs.unsqueeze(1) * (log_softmax_outputs.unsqueeze(1) - log_softmax_outputs.unsqueeze(0))).sum(dim=3)

            kl_matrix_ij = kl_ij.mean(dim=2)
            kl_matrix_ji = kl_ji.mean(dim=2)

            # print(f"KL_ij: {kl_ij}")

            diversity_loss = (-1) * ( kl_matrix_ij + kl_matrix_ji)

        elif config.DIVERSITY_TYPE == "cosine":
            # Flatten all outputs for cosine similarity calculation
            flat_outputs = torch.stack([out.view(out.size(0), -1) for out in outputs], dim=0)  # Shape: [N_generators, batch_size, features]

            # Normalize outputs along the feature dimension
            normalized_outputs = torch.nn.functional.normalize(flat_outputs, p=2, dim=2)  # Shape: [N_generators, batch_size, features]

            # Compute pairwise cosine similarity using matrix multiplication
            cosine_similarity_matrix = torch.einsum('gbi,gbj->gij', normalized_outputs, normalized_outputs)  # Shape: [N_generators, N_generators, batch_size]

            # Compute diversity loss (1 - cosine similarity) averaged over the batch
            diversity_loss = cosine_similarity_matrix.mean(dim=2)  # Shape: [N_generators, N_generators]

        else:
            raise ValueError(f"Invalid diversity type: {config.DIVERSITY_TYPE}")

        # Train each generator
        errors = [] 
        for i, gen in enumerate(self.generators):
            # consider the generator's individual performance WITH how seperate it's performance is relative to other generators

            # Mask out self-comparison and compute the sum
            diversity_loss_curr = diversity_loss[i].sum() - diversity_loss[i, i]

            # print(f"Generator {i} Diversity Loss: {diversity_loss_curr}")
            if config.VARY_REG:
                varied_reg = torch.pow(10, -torch.linspace(3, 8, len(self.generators))[i])
                # print(f"Generator {i} Regularization: {varied_reg}")
            gen_loss, just_gen_loss = gen.train_g(discriminator, real_batch, adv_batch[i], all_labels, kl_weight*diversity_loss_curr.detach(), varied_reg)

            errors.append(gen_loss)
            wandb.log({f"Gen. {i} Diversity Loss": kl_weight*diversity_loss_curr})
            wandb.log({f"Gen. {i} Generator Loss": just_gen_loss})
        
        # choose random generator to return
        # return max(errors).detach()

        # random_gen = torch.randint(0, len(self.generators), (1,))
        chosen_gen = torch.argmin(torch.stack(errors))
        
        return errors[chosen_gen].detach() + diversity_loss_curr, errors[chosen_gen].detach()

def noise(size, batch_size=config.BATCH_SIZE):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.rand(batch_size, *size) * 2 - 1)
    return n

def run():

    # Load data
    train_data = cifar10_data(train=True)# Create loader with data, so that we can iterate over it
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_data = cifar10_data(train=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    config.Z_SHAPE = test_data[0][0].shape

    # Initialize discriminator and generator(s)
    discriminator = DiscriminatorNet(config.Z_SHAPE[0])
    generator = GeneratorNetGroup(config.N_GENERATOR, config.Z_SHAPE[0])
    # mnist_classifier = Mnist_Classifier(28, 10)

    # Num batches
    num_batches = len(train_data_loader)

    num_epochs = config.EPOCHS
    g_losses = []
    d_losses = []
    g_test_acc = None
    d_test_acc = None
    for epoch in range(num_epochs):
        for n_batch, (real_batch, labels) in enumerate(train_data_loader):

            # noise_data = noise(config.Z_SHAPE, real_batch.size(0))
            adv_batch = [g + real_batch for g in generator(real_batch)]

            # Ensure adversarial batch is not nan or inf
            if any([torch.isnan(b).any() or torch.isinf(b).any() for b in adv_batch]):
                raise ValueError("Adversarial batch contains nan or inf values")

            # discriminator winning
            if g_test_acc != None and g_test_acc > d_test_acc and config.TRAINING_ALTERNATE:

                #1. Train Generator
                g_error, g_diversity = generator.train_g(discriminator, real_batch, adv_batch, labels)   

                #2.  Train Discriminator
                d_error, d_pred, loss_tup = discriminator.train_d(generator, real_batch, adv_batch, labels)

            # Generator winning or no data yet
            else:

                # 1. Train Discriminator
                d_error, d_pred, loss_tup = discriminator.train_d(generator, real_batch, adv_batch, labels)

                # 2. Train Generator      

                # Train gen until discriminator fails to a certain amount
                if config.TRAIN_GEN_MAX:
                    d_fail = 1
                    prev_d_fail = 0
                    counter = 0
                    while d_fail > 0.8 and abs(d_fail-prev_d_fail) > 0.001:
                        # Train gen
                        g_error, _ = generator.train_g(discriminator, real_batch, adv_batch, labels)
                        prev_d_fail = d_fail# Get discriminator error
                        adv_batch = [g + real_batch for g in generator(real_batch)]
                        if any([torch.isnan(b).any() or torch.isinf(b).any() for b in adv_batch]):
                            raise ValueError("Adversarial batch contains nan or inf values")
                        d_pred = discriminator(adv_batch[0])
                        d_fail = torch.sum(torch.argmax(d_pred, dim=1) == labels).float() / len(labels)
                        counter += 1
                        # print(f"Discriminator accuracy: {d_fail}")
                        

                    # if counter > 1:
                    #     print(f"Generator trained {counter} times")
                else:
                    g_error, g_diversity = generator.train_g(discriminator, real_batch, adv_batch, labels)

            # train mnist classifier on side
            # mnist_error = mnist_classifier.train_s(real_batch, labels)
        
            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)        

            wandb.log({"Generator Error": g_error, "Discriminator Error": d_error})

            # Display Progress every few batches
            if (n_batch) % config.BATCH_SIZE == 0:
                with torch.no_grad():   
                    # choose next test batch from test data loader. if end of data, reset. we can do performance on entire test set later.
                    try:
                        test_batch, labels = next(iter(test_data_loader)) 
                    except StopIteration:
                        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
                        test_batch, labels = next(iter(test_data_loader))

                    test_batch.to(config.DEVICE)
                    test_batch.requires_grad = True 


                    # Sample some noise and generate test images from the current batch
                    test_images_noise = generator(test_batch)

                    # ensure adv batch is not nan or inf
                    if any([torch.isnan(b).any() or torch.isinf(b).any() for b in test_images_noise]):
                        raise ValueError("Adversarial batch contains nan or inf values")
                    # print(test_images_noise[0], test_batch[0])
                    test_images = [g + test_batch for g in test_images_noise]

                    # print ball of noise
                    print([torch.mean(torch.linalg.norm(test_images_noise[i], ord=1, dim=1)) for i in range(len(test_images_noise))])

                    # Log Noise in wandb (default)
                    for i, img in enumerate(test_images_noise):
                        # normalize to 0-1
                        if torch.max(img) != torch.min(img):
                            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
                        else:
                            img = img - torch.min(img)
                        logger.log_images("Generated Noise {}".format(i), img, 100, epoch, n_batch, num_batches)

                    # Log Generated Images in wandb
                    for i, img in enumerate(test_images):
                        if torch.max(img) != torch.min(img):
                            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
                        else:
                            img = img - torch.min(img)
                        logger.log_images("Generated {}".format(i), img, 100, epoch, n_batch, num_batches)

                    # # Log Real Images in wandb
                    # for i, img in enumerate(test_batch):
                    #     logger.log_images("Gen input {}".format(i), img, 100, epoch, n_batch, num_batches)

                    # Display status Logs
                    logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred)

                    print("Disc Loss each type: ", loss_tup[0].item(), loss_tup[1].item())

                    wandb.log({"Gen Noise Max: ": torch.max(test_images_noise[0]), "Gen Noise Min: ": torch.min(test_images_noise[0])})
                    


                    # Log discriminator error on perturbed and unperturbed data       
                    discriminator.eval()    
                    unperturbed = discriminator(Variable(images_to_vectors(test_batch)))
                    perturbed = [discriminator(Variable(images_to_vectors(p))) for p in test_images]
                    # mnist_classifications = mnist_classifier(Variable(images_to_vectors(test_images[0])))

                    # calculate accuracy
                    unperturbed_accuracy = torch.sum(torch.argmax(unperturbed, dim=1) == labels).float() / len(labels)
                    perturbed_accuracy = [torch.sum(torch.argmax(p, dim=1) == labels).float() / len(labels) for p in perturbed]
                    # mnist_performance = torch.sum(torch.argmax(mnist_classifications, dim=1) == labels).float() / len(labels)

                    g_test_acc = min(perturbed_accuracy)
                    d_test_acc = unperturbed_accuracy

                    wandb_log = {"Unperturbed Accuracy": unperturbed_accuracy}
                    
                    if config.N_GENERATOR == 1:
                        wandb_log["Perturbed Accuracy"] = perturbed_accuracy[0]
                    else:
                        wandb_log["Min Perturbed Accuracy"] = min(perturbed_accuracy)
                        wandb_log["Max Perturbed Accuracy"] = max(perturbed_accuracy)

                    # wandb_log["Mnist Classifier Accuracy"] = mnist_performance
                        
                    wandb.log(wandb_log)

                # Not in no-grad because need gradients to calculate adversarial examples
                # Record fgsm error compared to perturbed data
                # ensure attack isnt changing params
                fgsm_perturbed_pred, _ = comparison.apply_fgsm_attack(discriminator, 0.03, test_batch, labels, CEloss)
                fgsm_perturbed_accuracy = torch.sum(torch.argmax(fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
                wandb.log({"FGSM Perturbed Accuracy": fgsm_perturbed_accuracy})

                # Record pgd error compared to perturbed data
                epsilon = 0.03
                alpha = 0.005
                n_iters = 100
                pgd_perturbed_pred, _ = comparison.apply_pgd_attack(discriminator, epsilon, alpha, n_iters, test_batch, labels, CEloss)
                pgd_perturbed_accuracy = torch.sum(torch.argmax(pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
                wandb.log({"PGD Perturbed Accuracy": pgd_perturbed_accuracy})

                # print("FGSM Perturbed Accuracy: ", fgsm_perturbed_accuracy)
                # print("PGD Perturbed Accuracy: ", pgd_perturbed_accuracy)

        d_losses.append(d_error)
        g_losses.append(g_error)
    

    # Save models
    utils.save_model(generator, config, name="generator_" + wandb.run.name, override=True)
    utils.save_model(discriminator, config, name="discriminator_" + wandb.run.name, override=True)

    wandb.finish()
    
if __name__ == "__main__":    

    # Initialize wandb
    wandb.init(
        project="DefendaGAN",
        config={
            "epochs": config.EPOCHS,
            "dataset":"CIFAR10",
        }
    )

    # Initialize logger
    logger = Logger(model_name='VGAN', data_name='CIFAR10')# Total number of epochs to train

    # Initialize Loss functions
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()

    #to run experiments, change configs here before running
    run()
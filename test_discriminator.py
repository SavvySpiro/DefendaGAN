import utils
import network
import comparison
import torch
import dataset
import config
from torch import nn, optim


# exported models are manually moved to a directory
state_dict_path = "/home/hanna/Documents/DefendaGAN/fancy-sea-80/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_easiest_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth"
model = utils.load_model(network.DiscriminatorNet(), state_dict_path)

######################################################

# Load data
all_data = dataset.mnist_data()# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(all_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
# Num batches
num_batches = len(data_loader)

######################################################
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for n_batch, (real_batch, labels) in enumerate(data_loader):
    output = model(real_batch)
    # calculate accuracy
    accuracy = torch.sum(torch.argmax(output, dim=1) == labels).float() / len(labels)

    CEloss = nn.CrossEntropyLoss()
    epsilon = 0.1
    # Record fgsm error compared to perturbed data
    fgsm_perturbed_pred, _ = comparison.apply_fgsm_attack(model, epsilon, real_batch, labels, CEloss)
    fgsm_perturbed_accuracy = torch.sum(torch.argmax(fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
    print("FGSM Perturbed Accuracy: ", fgsm_perturbed_accuracy)

    for target_class in classes:
        targeted_fgsm_perturbed_pred, _ = comparison.apply_targeted_fgsm_attack(model, epsilon, real_batch, labels, CEloss, target_class)
        targeted_fgsm_perturbed_accuracy = torch.sum(torch.argmax(targeted_fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
        print(f"targeted FGSM Perturbed Accuracy for class {target_class}: ", targeted_fgsm_perturbed_accuracy)

    # Record pgd error compared to perturbed data
    epsilon = 0.1
    alpha = 0.005
    n_iters = 50
    pgd_perturbed_pred, _ = comparison.apply_pgd_attack(model, epsilon, alpha, n_iters, real_batch, labels, CEloss)
    pgd_perturbed_accuracy = torch.sum(torch.argmax(pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
    print("PGD Perturbed Accuracy: ", pgd_perturbed_accuracy)
    
    for target_class in classes:
        targeted_pgd_perturbed_pred, _ = comparison.apply_targeted_fgsm_attack(model, epsilon, real_batch, labels, CEloss, target_class)
        targeted_pgd_perturbed_accuracy = torch.sum(torch.argmax(targeted_pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
        print(f"targeted PGD Perturbed Accuracy for class {target_class}: ", targeted_pgd_perturbed_accuracy)

    break

######################################################

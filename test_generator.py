import visualize as vis
import utils
import network
import comparison
import torch
import dataset
import config
from torch import nn, optim
from torch.autograd.variable import Variable

def noise(size, batch_size=config.BATCH_SIZE):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.rand(batch_size, *size) * 2 - 1)
    return n

if __name__ == "__main__":    
    # load baseline victim model
    state_dict_path = "models/baseline_mnist.pth"
    criterion = nn.CrossEntropyLoss() # cross entropy loss
    victim_baseline = utils.load_model(network.mnist_classifier(28, 128, 10), state_dict_path)

    # load an attacking generator
    # state_dict_path = "/home/hanna/Documents/DefendaGAN/generator_BATCH_SIZE_128_DEVICE_cuda_DISABLE_ATTACK_True_EPOCHS_10_GENERATOR_LR_0.0002_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_0.01_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth"
    state_dict_path="/home/hanna/Documents/DefendaGAN/to test/mod/generator_BATCH_SIZE_128_DEVICE_cuda_DISABLE_ATTACK_False_EPOCHS_50_GENERATOR_LR_0.0002_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_0.01_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_False_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth"
    attacker = utils.load_model(network.GeneratorNetGroup(1), state_dict_path)
    
    # load a discriminator
    state_dict_path = "/home/hanna/Documents/DefendaGAN/models/alt and gen loss/alt/rural-capybara-158 (alt, even)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_True_Z_SHAPE_(16, 21, 21)_.pth"
    discrim_model = utils.load_model(network.DiscriminatorNet(), state_dict_path)

    # Load data
    train_data = dataset.mnist_data(train=True)# Create loader with data, so that we can iterate over it
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_data = dataset.mnist_data(train=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    # Num batches
    num_batches = len(train_data_loader)

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for n_batch, (real_batch, labels) in enumerate(test_data_loader):
        id = 2
        noise_data = noise(config.Z_SHAPE, real_batch.size(0))
        adv_batch = [g + real_batch for g in attacker(noise_data)]
        
        vis.vis_digit(victim_baseline, adv_batch[0][id], labels[id], text="adv")
        vis.vis_digit(victim_baseline, real_batch[id], labels[id], text="real")

        predictions_adv = victim_baseline(adv_batch[0])
        predictions_real = victim_baseline(real_batch)

        accuracy_adv = torch.sum(torch.argmax(predictions_adv, dim=1) == labels).float() / len(labels)
        accuracy_real = torch.sum(torch.argmax(predictions_real, dim=1) == labels).float() / len(labels)

        print(f"attacked accuracy: {accuracy_adv}")
        print(f"real accuracy: {accuracy_real}")
        break
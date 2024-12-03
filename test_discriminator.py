import utils
import network
import comparison
import torch
import dataset
import config
from torch import nn, optim

def run_attacks(model, real_batch, labels):
    output = model(real_batch)
    # calculate accuracy
    accuracy = torch.sum(torch.argmax(output, dim=1) == labels).float() / len(labels)
    print("Accuracy on original images: ", accuracy)

    ##################################
    # Record carlini-wagner error compared to perturbed data
    cw_perturbed_pred, _ = comparison.apply_cw_attack(model, real_batch, labels, c=1, kappa=10, steps=1000, lr=0.01)
    cw_perturbed_accuracy = torch.sum(torch.argmax(cw_perturbed_pred, dim=1) == labels).float() / len(labels)
    print("Carlini-Wagner Perturbed Accuracy: ", cw_perturbed_accuracy)

    #############################################
    CEloss = nn.CrossEntropyLoss()
    epsilon = 0.3
    # Record fgsm error compared to perturbed data
    fgsm_perturbed_pred, _ = comparison.apply_fgsm_attack(model, epsilon, real_batch, labels, CEloss)
    fgsm_perturbed_accuracy = torch.sum(torch.argmax(fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
    print("FGSM Perturbed Accuracy: ", fgsm_perturbed_accuracy)

    # for target_class in classes:
    #     targeted_fgsm_perturbed_pred, _ = comparison.apply_targeted_fgsm_attack(model, epsilon, real_batch, labels, CEloss, target_class)
    #     targeted_fgsm_perturbed_accuracy = torch.sum(torch.argmax(targeted_fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
    #     print(f"targeted FGSM Perturbed Accuracy for class {target_class}: ", targeted_fgsm_perturbed_accuracy)

    # Record pgd error compared to perturbed data
    epsilon = 0.3
    alpha = 0.005
    n_iters = 100
    pgd_perturbed_pred, _ = comparison.apply_pgd_attack(model, epsilon, alpha, n_iters, real_batch, labels, CEloss)
    pgd_perturbed_accuracy = torch.sum(torch.argmax(pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
    print("PGD Perturbed Accuracy: ", pgd_perturbed_accuracy)
    
    # for target_class in classes:
    #     targeted_pgd_perturbed_pred, _ = comparison.apply_targeted_fgsm_attack(model, epsilon, real_batch, labels, CEloss, target_class)
    #     targeted_pgd_perturbed_accuracy = torch.sum(torch.argmax(targeted_pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
    #     print(f"targeted PGD Perturbed Accuracy for class {target_class}: ", targeted_pgd_perturbed_accuracy)


if __name__ == "__main__":    
    # load baseline
    state_dict_path = "models/baseline_mnist.pth"
    criterion = nn.CrossEntropyLoss() # cross entropy loss
    model_baseline = utils.load_model(network.mnist_classifier(28, 128, 10), state_dict_path)

    # load model
    # all_models = ["models/alt and gen loss/alt/divine-leaf-218 (alt, target)/discriminator_BATCH_SIZE_128_DISABLE_ATTACK_False_EPOCHS_50_GEN_LOSS_target_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_True_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/alt/exalted-dream-165 (alt, easiest)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_easiest_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_True_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/alt/restful-frost-169 (alt, easiest_exp)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_easiest_exp_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_True_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/alt/rural-capybara-158 (alt, even)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_True_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/no_alt/drawn-forest-70 (no_alt_easiest)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_easiest_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/no_alt/efficient-vortex-128 (no_alt, easiest)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_easiest_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/no_alt/glowing-mountain-157 (no_alt, easiest_exp)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_easiest_exp_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/no_alt/happy-dawn-68 (no_alt, even)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth", 
    #               "models/alt and gen loss/no_alt/pious-sound-161 (no_alt, target)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_target_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth"]

    all_models = ["/home/hanna/Documents/DefendaGAN/discriminator_BATCH_SIZE_128_DEVICE_cuda_DISABLE_ATTACK_False_EPOCHS_10_GENERATOR_LR_0.0002_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_0.01_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth",
                  "discriminator_BATCH_SIZE_128_DEVICE_cuda_DISABLE_ATTACK_True_EPOCHS_10_GENERATOR_LR_0.0002_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_0.01_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_False_Z_SHAPE_(16, 21, 21)_.pth"]
    # state_dict_path = "/home/hanna/Documents/DefendaGAN/models/alt and gen loss/alt/rural-capybara-158 (alt, even)/discriminator_BATCH_SIZE_128_EPOCHS_50_GEN_LOSS_even_INPUT_TYPE_noise_KL_WEIGHT_100_KL_WEIGHT_SCHEDULE_1_N_GENERATOR_1_REGULARIZATION_True_TRAINING_ALTERNATE_True_Z_SHAPE_(16, 21, 21)_.pth"
    # model = utils.load_model(network.DiscriminatorNet(), state_dict_path)

    ######################################################
    # Load data
    train_data = dataset.mnist_data(train=True)# Create loader with data, so that we can iterate over it
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_data = dataset.mnist_data(train=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    # Num batches
    num_batches = len(train_data_loader)

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for model_path in all_models:
      model = utils.load_model(network.DiscriminatorNet(), model_path)
      for n_batch, (real_batch, labels) in enumerate(test_data_loader):
          run_attacks(model_baseline, real_batch, labels)
          print(model_path)
          run_attacks(model, real_batch, labels)
          print("==============================")
          break

    ######################################################

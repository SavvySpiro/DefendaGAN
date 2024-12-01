import utils
import network
import comparison
import torch
import dataset
import config
from torch import nn, optim
import torchattacks

# exported models are manually moved to a directory
state_dict_path = "models/baseline_mnist.pth"

# Load data
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
  
mnist_dim = 28 # dimension of mnist images
layer_size = 128
classes = 10 # digits 0-9
model = mnist_classifier(mnist_dim, layer_size, classes)
criterion = nn.CrossEntropyLoss() # cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001) # adam optimizer

model_baseline = utils.load_model(mnist_classifier(mnist_dim, layer_size, classes), state_dict_path)
######################################################

all_data = dataset.mnist_data()# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(all_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
# Num batches
num_batches = len(data_loader)

######################################################
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for n_batch, (real_batch, labels) in enumerate(data_loader):
    output = model_baseline(real_batch)
    # calculate accuracy
    accuracy = torch.sum(torch.argmax(output, dim=1) == labels).float() / len(labels)
    print("Accuracy on original images: ", accuracy)
    # ##################################
    # # Record carlini-wagner error compared to perturbed data
    # cw_perturbed_pred, _ = comparison.apply_cw_attack(model_baseline, real_batch, labels, c=1e-4, kappa=0, steps=100, lr=0.01, )
    # cw_perturbed_accuracy = torch.sum(torch.argmax(cw_perturbed_pred, dim=1) == labels).float() / len(labels)
    # print("Carlini-Wagner Perturbed Accuracy: ", accuracy)

    CEloss = nn.CrossEntropyLoss()
    epsilon = 0.1
    # Record fgsm error compared to perturbed data
    fgsm_perturbed_pred, _ = comparison.apply_fgsm_attack(model_baseline, epsilon, real_batch, labels, CEloss)
    fgsm_perturbed_accuracy = torch.sum(torch.argmax(fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
    print("FGSM Perturbed Accuracy: ", fgsm_perturbed_accuracy)

    # for target_class in classes:
    #     targeted_fgsm_perturbed_pred, _ = comparison.apply_targeted_fgsm_attack(model_baseline, epsilon, real_batch, labels, CEloss, target_class)
    #     targeted_fgsm_perturbed_accuracy = torch.sum(torch.argmax(targeted_fgsm_perturbed_pred, dim=1) == labels).float() / len(labels)
    #     print(f"targeted FGSM Perturbed Accuracy for class {target_class}: ", targeted_fgsm_perturbed_accuracy)

    # # Record pgd error compared to perturbed data
    # epsilon = 0.1
    # alpha = 0.005
    # n_iters = 50
    # pgd_perturbed_pred, _ = comparison.apply_pgd_attack(model_baseline, epsilon, alpha, n_iters, real_batch, labels, CEloss)
    # pgd_perturbed_accuracy = torch.sum(torch.argmax(pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
    # print("PGD Perturbed Accuracy: ", pgd_perturbed_accuracy)
    
    # for target_class in classes:
    #     targeted_pgd_perturbed_pred, _ = comparison.apply_targeted_fgsm_attack(model_baseline, epsilon, real_batch, labels, CEloss, target_class)
    #     targeted_pgd_perturbed_accuracy = torch.sum(torch.argmax(targeted_pgd_perturbed_pred, dim=1) == labels).float() / len(labels)
    #     print(f"targeted PGD Perturbed Accuracy for class {target_class}: ", targeted_pgd_perturbed_accuracy)

    break
######################################################

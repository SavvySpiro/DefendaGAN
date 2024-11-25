import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from matplotlib import pyplot as plt

# Set up the FGSM attack
# 1. Get predicted class of nominal image, x
# 2. Calculate gradient of loss w.r.t. input image, – what loss to use?
# 3. Use as direction to perturb image, scale by 𝜖 – why sign?
# ∇x J(x 0; θ)
# sign( ∇x J(x 0; θ))

def apply_fgsm_attack(model, epsilon, original_images, labels, criterion):
    model.eval()  # Set the model to evaluation mode
    original_images.requires_grad = True  # Enable gradient computation on the input image

    # Get the model's predictions for the original image
    output = model(original_images)  # Forward pass through the model
    loss = criterion(output, labels)  # Compute the loss

    # Zero all gradients before backward pass (we don't want gradients w.r.t the model's params)
    model.zero_grad()  # Reset gradients for model parameters

    # Backpropagate to compute the gradient w.r.t the input image
    loss.backward()  # Compute gradients on the input image

    data_grad = original_images.grad.detach()  # Get gradients w.r.t the input image

    # Apply the FGSM attack (perturb the input image)
    sign_data_grad = data_grad.sign()  # Get the sign of the gradient
    perturbed_image = original_images + epsilon * sign_data_grad  # Perturb the image
    perturbed_data = torch.clamp(perturbed_image, 0, 1)  # Clamp to valid pixel range

    with torch.no_grad():
        perturbed_predictions = model(perturbed_data)

    return perturbed_predictions, perturbed_data
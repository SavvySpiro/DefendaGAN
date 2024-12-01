import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import torchattacks

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


# Set up targeted FGSM attack
# 1. Calculate gradient of loss w.r.t. input image, – what loss?
# 2. Use as direction to perturb image, scale by 𝜖
# Define the main function to run the targeted attack
def apply_targeted_fgsm_attack(model, epsilon, original_images, labels, criterion, target_class):
    model.eval()

    original_images.requires_grad = True

    # get the predicted class of nominal image
    output = model(original_images) # run images through the model, forward pass
    # create the target class you want to mislead the classifier to identify
    target_tensor = torch.full_like(labels, target_class, dtype=torch.long)
    loss = criterion(output, target_tensor) # compute loss

    model.zero_grad() # zero gradients
    loss.backward() # to get gradient backward pass
    data_grad = original_images.grad.detach()

    sign_data_grad = data_grad.sign() # get gradient elements with sign
    perturbed_image = original_images - epsilon * sign_data_grad # use the gradient as direction to perturb the image
    # ^ targeted_fgsm_attack computes the perturbed image by applying the FGSM formula to minimize the
    # loss for a specific target class: perturbed_image = image - epsilon * sign(gradient).
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # clamp the pixels to make sure the image remains valid
    
    with torch.no_grad():
        perturbed_predictions = model(perturbed_image)

    return perturbed_predictions, perturbed_image


def apply_pgd_attack(model, epsilon, alpha, num_iter, original_images, labels, criterion):
    model.eval()
    
    # clone image and ensure it requires gradients
    perturbed_image = original_images.clone().detach().requires_grad_(True)
    
    for i in range(num_iter):
        # forward pass
        output = model(perturbed_image)
        loss = criterion(output, labels)
        
        # backward pass
        model.zero_grad()
        loss.backward()
        
        # get the gradient sign
        grad_sign = perturbed_image.grad.sign()
        
        # step in the direction of the gradient
        perturbed_image = perturbed_image + alpha * grad_sign
        
        # clip the perturbed image to be within the epsilon-ball and valid pixel range
        perturbed_image = torch.clamp(perturbed_image, original_images - epsilon, original_images + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Assuming pixel values are in [0, 1]
        
        # detach the perturbed image and re-enable gradient computation
        perturbed_image = perturbed_image.detach()
        perturbed_image.requires_grad = True
    
        with torch.no_grad():
            perturbed_predictions = model(perturbed_image)

    return perturbed_predictions, perturbed_image

def apply_targeted_pgd_attack(model, epsilon, alpha, n_iters, original_images, labels, criterion, target_class):
    model.eval()
    
    # clone image and ensure it requires gradients
    perturbed_image = original_images.clone().detach().requires_grad_(True)

    # get predictions for the original images
    output = model(original_images)
    target_tensor = torch.full_like(labels, target_class, dtype=torch.long)

    # initialize perturbed image
    perturbed_image = original_images.clone().detach().requires_grad_(True)  

    for it in range(n_iters):
        output = model(perturbed_image)  # forward pass
        loss = criterion(output, target_tensor)  # get loss for the target class
        model.zero_grad()  # zero gradients
        loss.backward()  # backpropagate to get gradients

        data_grad = perturbed_image.grad.detach()  # compute gradient
        perturbed_image = perturbed_image - alpha * data_grad.sign()  # update image in gradient direction
        # clip the perturbed image to be within the epsilon-ball and valid pixel range
        perturbed_image = torch.clamp(original_images + torch.clamp(perturbed_image - original_images, -epsilon, epsilon), 0, 1)
        
        # detach the perturbed image and re-enable gradient computation
        perturbed_image = perturbed_image.detach().requires_grad_(True)

    with torch.no_grad():
        perturbed_predictions = model(perturbed_image)

    return perturbed_predictions, perturbed_image

def apply_perturb_images_const(model, original_images, c=1.2):
    offset = c * torch.ones_like(original_images) # create constant offset
    perturbed_images = original_images + offset # apply constant pertubation
    perturbed_data  = torch.clamp(perturbed_images, 0, 1) # clamp values to be in range

    with torch.no_grad():
        perturbed_predictions = model(perturbed_data)

    return perturbed_predictions, perturbed_data

def apply_perturb_images_rand_noise(model, noise_mag, original_images):
    noise = noise_mag * torch.randn_like(original_images) # generate random noise
    rand_perturbed_images = original_images + noise # apply random noise
    perturbed_data = torch.clamp(rand_perturbed_images, 0, 1) # clamp values
    with torch.no_grad():
        perturbed_predictions = model(perturbed_data)

    return perturbed_predictions, perturbed_data


def apply_cw_attack(model, original_images, labels, c, kappa, steps, lr):
    model.eval()  # Set the model to evaluation mode
    original_images.requires_grad = True  # Enable gradient computation on the input image

    atk = torchattacks.CW(model, c, kappa, steps, lr)
    # Generate adversarial examples 
    perturbed_data = atk(original_images, labels)

    with torch.no_grad():
        perturbed_predictions = model(perturbed_data)

    return perturbed_predictions, perturbed_data
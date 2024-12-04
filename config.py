
import torch

# Configurations for the GAN
TRAINING_ALTERNATE = True # True or False, where it alternates if one of them is winning/losing
REGULARIZATION = True # True or False, where true means similar looking images, false means diverse looking images
REG_STRENGTH = 1e-12 # Regularization strength
BATCH_SIZE = 128 # Batch sizes
INPUT_TYPE = "noise" # "noise" or "image", where noise is random noise and image is the image as input to generator TODO
N_GENERATOR = 1 # Number of generators
GENERATOR_LR = 0.001 # Learning rate for the generators
DISCRIMINATOR_LR = 0.001 # Learning rate for the discriminator
Z_SHAPE = (1,28,28) # Don't change!
KL_WEIGHT = 0.0001 # KL weight, the bigger the number the more diverse the generators are forced to be
KL_WEIGHT_SCHEDULE = 1 # KL weight schedule. use 1 for no schedule #TODO: Implement schedule
DIVERSITY_TYPE = "cosine" # "KL" or cosine where kl is the KL divergence and entropy is the entropy
VARY_REG = True # True or False, where true means the regularization strength varies by number of gens
GEN_LOSS = "even" # even - true is 0, others equal probability
                # target - true is 0, some other target is 1
                # easiest - true is 0, some other target is 1, but the easiest one
                # easiest_exp
EPOCHS = 50
DISABLE_ATTACK = False # True or False, where true means the discriminator is disabled
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
'''
List of experimental variables to test/change:
1. Training in the same loop or alternating using TRAINING_ALTERNATE
2. Regularization or not using REGULARIZATION (aka similar looking images or diverse looking images)
3. Input type as noise or image using INPUT_TYPE
4. Number of generators using N_GENERATOR
5. KL weight using KL_WEIGHT which chooses how diverse the generators must be (0 is not diverse, 1 is diverse)
6. KL weight schedule using KL_WEIGHT_SCHEDULE which schedules the KL weight to increase over time (start with similar gens and diverge)
7. 

Issues:
1. Fix baseline: why is discriminator good when we've disabled the attack?
2. Alternating doesn't make that much a different? ^ related
3. Generative loss changes don't make that much a difference?
'''
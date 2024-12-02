


# Configurations for the GAN
TRAINING_ALTERNATE = False # True or False, where it alternates if one of them is winning/losing
REGULARIZATION = True # True or False, where true means similar looking images, false means diverse looking images
BATCH_SIZE = 128 # Batch size
INPUT_TYPE = "noise" # "noise" or "image", where noise is random noise and image is the image as input to generator TODO
N_GENERATOR = 3 # Number of generators
GENERATOR_LR = 0.0002 # Learning rate for the generators
Z_SHAPE = (16, 21, 21) # Don't change!
KL_WEIGHT = 0.01 # KL weight, the bigger the number the more diverse the generators are forced to be
KL_WEIGHT_SCHEDULE = 1 # KL weight schedule. use 1 for no schedule #TODO: Implement schedule
GEN_LOSS = "easiest" # even - true is 0, others equal probability
                # target - true is 0, some other target is 1
                # easiest - true is 0, some other target is 1, but the easiest one
EPOCHS = 50
DIVERSITY_TYPE = "cosine" # "kl" or "cosine", where kl is KL divergence and cosine is cosine similarity
'''
List of experimental variables to test/change:
1. Training in the same loop or alternating using TRAINING_ALTERNATE
2. Regularization or not using REGULARIZATION (aka similar looking images or diverse looking images)
3. Input type as noise or image using INPUT_TYPE
4. Number of generators using N_GENERATOR
5. KL weight using KL_WEIGHT which chooses how diverse the generators must be (0 is not diverse, 1 is diverse)
6. KL weight schedule using KL_WEIGHT_SCHEDULE which schedules the KL weight to increase over time (start with similar gens and diverge)
7. 
'''
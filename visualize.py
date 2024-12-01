    
import matplotlib.pyplot as plt
import torch

def vis_digit(model, data, label, text=" "):
    outputs = model(data)
    predicted = torch.max(outputs.data, 1, keepdim=True)[1]
    fig, ax = plt.subplots(1)
    ax.imshow(data[0].detach(), cmap='gray', interpolation='none')
    ax.text(5, 1.5, f'predicted: {predicted.numpy()[0]}, actual: {label} --> {text}', bbox={'facecolor': 'white', 'pad': 10})
    plt.show()
    plt.savefig(f'test_{text}.png') 

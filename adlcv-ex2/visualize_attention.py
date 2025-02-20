import torch
from vit import ViT
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import random
import torch

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import torch
import numpy as np


def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes 
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, trainset, testset



def compute_rollout_attention(attention_maps_tensor, discard_ratio=0.9):
    """
    Compute the rollout attention from attention maps across blocks and heads.
    
    Args:
        attention_maps_tensor (torch.Tensor): Attention maps with shape [num_blocks, num_heads, num_tokens, num_tokens].
        discard_ratio (float): The fraction of low attention values to discard.
        
    Returns:
        torch.Tensor: Final attention map with shape [num_tokens - 1] (class token to all patches).
    """
    # Move everything to the CPU
    attention_maps_tensor = attention_maps_tensor.cpu()
    
    # Step 1: Average attention across all heads for each block
    averaged_attention_maps = attention_maps_tensor.mean(dim=1)  # Shape: [num_blocks, num_tokens, num_tokens]
    
    # Step 2: Initialize the result as an identity matrix (self-attention)
    result = torch.eye(averaged_attention_maps.size(-1), device='cpu')  # Shape: [num_tokens, num_tokens]
    
    # Step 3: Iterate over blocks and recursively propagate attention
    for attention in averaged_attention_maps:
        # Flatten and discard low attention values (out-of-place operation)
        flat = attention.flatten()  # Flatten to a 1D tensor
        _, indices = torch.topk(flat, int(flat.size(0) * discard_ratio), largest=False)  # Get indices of low values
        attention_clone = attention.clone()  # Avoid in-place modifications
        attention_clone.view(-1)[indices] = 0  # Set discarded values to zero
        
        # Add identity matrix (self-loops)
        I = torch.eye(attention_clone.size(-1), device='cpu')
        attention_clone = (attention_clone + I) / 2
        
        # Normalize rows
        attention_clone = attention_clone / attention_clone.sum(dim=-1, keepdim=True)
        
        # Multiply recursively with the cumulative result
        result = torch.matmul(attention_clone, result)
    
    # Step 4: Extract the class token's attention to all patches
    cls_token_attention = result[0, 1:]  # Exclude the class token itself
    cls_token_attention = cls_token_attention / cls_token_attention.max()  # Normalize to [0, 1]
    
    return cls_token_attention


def visualize_attention_on_image(attention, image):
    """
    Visualize the attention map overlaid on an image.
    
    Args:
        attention (torch.Tensor): Output from `compute_rollout_attention`, shape [64].
        image (torch.Tensor): Input image tensor, shape [3, 32, 32].
    """
    # Step 1: Reshape the attention map to 8x8
    attention_map = attention.reshape(8, 8)  # Shape: [8, 8]
    
    # Step 2: Upscale the attention map to 32x32 by duplicating values
    attention_map_32x32 = np.repeat(np.repeat(attention_map, 4, axis=0), 4, axis=1)  # Shape: [32, 32]
    
    # Step 3: Normalize attention map to range [0, 1] (just in case)
    attention_map_32x32 = attention_map_32x32 / attention_map_32x32.max()
    
    # Step 4: Apply the attention to the image (scale pixel intensities)
    image = image.cpu().numpy()  # Convert image to numpy array, shape [3, 32, 32]
    image = np.transpose(image.squeeze(), (1, 2, 0))  # Change shape to [32, 32, 3] for visualization
    attention_applied_image = image * attention_map_32x32[:, :, None]  # Apply attention to all color channels
    
    # Step 5: Plot the original image and the attention-overlaid image
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    # Attention Overlaid Image
    plt.subplot(1, 2, 2)
    plt.imshow(attention_applied_image)
    plt.title("Image with Attention Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming `attention_output` is the result of `compute_rollout_attention`
# and `image_tensor` is a [3, 32, 32] tensor
# visualize_attention_on_image(attention_output, image_tensor)

# Example usage:
# Assuming `attention_output` is the result of `compute_rollout_attention`
# visualize_attention_as_image(attention_output)


def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1
         
    ):

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    # Assuming the same model architecture as defined above
    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes, plot_attention=True)

    # Load the trained weights
    model.load_state_dict(torch.load('model.pth'))

    # Move the model to the appropriate device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()
    for image, label in test_iter:
        if torch.cuda.is_available():
            image, label = image.to('cuda'), label.to('cuda')
        
        out, attention_maps = model(image[0].unsqueeze(dim=0))
        attention_maps_tensor = torch.stack(attention_maps)
        
        attention_tensor = compute_rollout_attention(attention_maps_tensor)
        attention_array = attention_tensor.detach().numpy()

        visualize_attention_on_image(attention_array, image[0].unsqueeze(dim=0).cpu())
    print("Trained model loaded successfully!")

if __name__ == "__main__":
    main()


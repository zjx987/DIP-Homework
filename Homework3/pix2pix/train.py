import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from FCN_network import Discriminator
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, discriminator, dataloader, optimizer_g, optimizer_d, criterion_g, criterion_d, device, epoch, num_epochs):
    model.train()
    discriminator.train()
    running_loss_g = 0.0
    running_loss_d = 0.0
    lambda_L1=100

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # ====================
        # Train Discriminator
        # ====================
        optimizer_d.zero_grad()

        # Generate fake images
        fake_images = model(image_rgb)

        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_images, 'train_results', epoch)

        # Real images label: 1, Fake images label: 0
        real_labels = torch.ones(image_rgb.size(0), 1).to(device)
        fake_labels = torch.zeros(image_rgb.size(0), 1).to(device)

        # Discriminator loss on real images
        real_images = torch.cat((image_rgb, image_semantic), 1)  # Concatenate real images
        real_output = discriminator(real_images)
        loss_d_real = criterion_d(real_output.squeeze(dim=1).squeeze(dim=1), real_labels)

        # Discriminator loss on fake images
        fake_images_concat = torch.cat((image_rgb, fake_images.detach()), 1)  # Detach fake images from generator's graph
        fake_output = discriminator(fake_images_concat)
        loss_d_fake = criterion_d(fake_output.squeeze(dim=1).squeeze(dim=1), fake_labels)

        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # ====================
        # Train Generator
        # ====================
        optimizer_g.zero_grad()

        # Generator loss: Adversarial loss + L1 loss
        fake_images_concat = torch.cat((image_rgb, fake_images), 1)
        output_g = discriminator(fake_images_concat)
        loss_g_adversarial = criterion_d(output_g.squeeze(dim=1).squeeze(dim=1), real_labels)  # Generator tries to fool the discriminator
        loss_g_l1 = criterion_g(fake_images, image_semantic)  # L1 loss for image consistency

        loss_g_total = loss_g_adversarial +  lambda_L1 * loss_g_l1
        loss_g_total.backward()
        optimizer_g.step()

        # Update running losses
        running_loss_g += loss_g_total.item()
        running_loss_d += loss_d.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss G: {loss_g_total.item():.4f}, Loss D: {loss_d.item():.4f}')

    print(f'Epoch [{epoch + 1}/{num_epochs}] - Generator Loss: {running_loss_g / len(dataloader):.4f}, Discriminator Loss: {running_loss_d / len(dataloader):.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list_city.txt')
    val_dataset = FacadesDataset(list_file='val_list_city.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)
    criterion_g = nn.L1Loss()  # Generator L1 loss
    criterion_d = nn.BCELoss()
    optimizer_g = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_g = StepLR(optimizer_g, step_size=200, gamma=0.2)
    scheduler_d = StepLR(optimizer_d, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(model, discriminator, train_loader, optimizer_g, optimizer_d, criterion_g, criterion_d, device, epoch, num_epochs)
        validate(model, val_loader, criterion_g, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_g.step()
        scheduler_d.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()

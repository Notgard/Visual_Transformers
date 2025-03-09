import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as V
from datetime import datetime

def prediction_accuracy(model, data, device, flatten=True):
    correct = 0
    total = 0
    if device.type == "cuda":
        model = model.to(device)
    for image, label in data:
        image, label = image.to(device), label.to(device)
        
        if flatten:
            image = torch.reshape(image, (-1, 3*32*32))  
        
        output = model(image)
        _, pred_label = torch.max(output, dim=1)
        
        # Update total and correct counts
        total += label.size(0)
        correct += (pred_label == label).sum().item()
    
    accuracy = correct / total
    print(f'Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2f}')
    return accuracy

def fit_one_cycle(model, train_loader, optimizer, eindex, writer, device, flatten=True, size=None, log_freq=20, move_batch=True):
    running_loss = 0.0
    last_loss = 0.0
    i = 0
    
    data = train_loader
    if size is not None:
        print(f"Truncating dataset to {size} samples")
        data = []
        for batch in train_loader:
            if len(data) == size:
                break
            data.append(batch)
            
    num_batches = len(train_loader)
    log_interval = max(1, num_batches // log_freq)

    model.train(True)  # Set the model to training mode
    
    for image, label in tqdm(data, desc="Training", leave=True):
        optimizer.zero_grad()
        if move_batch:
            image, label = image.to(device), label.to(device)
        if flatten:
            image = torch.reshape(image, (-1, 3*32*32))
        
        output = model(image)
        
        probs = output.float()

        loss = F.cross_entropy(probs, label)
                
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % log_interval == (log_interval - 1):
            last_loss = running_loss / log_interval
            #print(print('  batch {} loss: {}'.format(i + 1, last_loss)))
            tb_x = eindex * len(train_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
        i += 1

    return last_loss
        
def training_and_validation_loop(model, train_data, test_data, epochs, writer, device, opt, timestamp, flatten=True, move_batch=True):
    best_vloss = 1_000_000.
    
    if device.type == "cuda":
        model = model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        model.train(True)
        avg_loss = fit_one_cycle(model, train_data, opt, epoch, writer, device, flatten)
        acc = prediction_accuracy(model, test_data, device, flatten=flatten)
        print(f"Accuracy: {acc * 100}% ({acc})\nLoss: {avg_loss}")

        writer.add_scalar('Accuracy/Test', acc, epoch + 1)
        
        i = 0
        
        running_vloss = 0.0
        model.eval()
        
        with torch.no_grad():
            for validation_data in tqdm(test_data, desc="Validation", leave=True):
                vimages, vlabels = validation_data
                if move_batch:
                    vimages, vlabels = vimages.to(device), vlabels.to(device)
                
                if flatten:
                    vimages = torch.reshape(vimages, (-1, 3*32*32))
                
                validation_output = model(vimages)
                validation_loss = F.cross_entropy(validation_output, vlabels)
                running_vloss += validation_loss
                
                i += 1
        
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        
        writer.flush()
        
        if avg_loss < best_vloss:
            best_vloss = avg_loss
            print("Saving model")
            model_path = 'saved_models/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

def move_dataloader_to_device(dataloader, device):
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
    return dataloader

def test_and_display_random_images(model, dataset, num_images=8, class_labels=None, device='cpu'):
    model.eval()
    indices = random.sample(range(len(dataset)), num_images)  # Select random indices
    images, true_labels, predicted_labels = [], [], []
    
    with torch.no_grad(): #avoid backpropagation and being computationally expensive
        for idx in indices:
            image, label = dataset[idx]
            images.append(image)
            true_labels.append(label)
            
            # Prepare the image for the model
            input_image = image.unsqueeze(0).to(device)
            output = model(input_image)  # Forward pass
            predicted_class = output.argmax(dim=1).item()
            predicted_labels.append(predicted_class)

    # Max images per row
    num_cols = 10
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # Plot the images in a grid
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3 * num_rows))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        img = V.to_pil_image(images[i])
        ax.imshow(img)
        true_label = true_labels[i]
        pred_label = predicted_labels[i]
        true_name = class_labels[true_label] if class_labels else str(true_label)
        pred_name = class_labels[pred_label] if class_labels else str(pred_label)
        ax.set_title(f"True: {true_name}\nPred: {pred_name}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
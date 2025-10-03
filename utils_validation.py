import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

def plot_class_distribution(dataset):
    """Plot distribusi kelas dalam dataset"""
    labels = [label for _, label in dataset.data]
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=unique, y=counts)
    plt.title('Distribusi Kelas')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Sampel')
    plt.show()
    return dict(zip(unique, counts))

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return cm

def check_model_predictions(model, dataloader, device, num_samples=5):
    """Cek prediksi model pada beberapa sampel"""
    model.eval()
    all_images = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_images.extend(images.cpu())
            all_labels.extend(labels)
            all_preds.extend(preds.cpu())
            
            if len(all_images) >= num_samples:
                break
    
    # Plot hasil
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        img = all_images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = img.clip(0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {all_labels[i]}\nPred: {all_preds[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
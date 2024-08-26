import os
import time
import logging
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy
import timm
from transformers import ViTFeatureExtractor
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchsummary import summary

torch.cuda.empty_cache()

log_file = 'mobilevit_xx_small_38.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

def log_system_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    gpu_memory = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0
    logging.info(f"CPU usage: {cpu_usage}%")
    logging.info(f"Memory usage: {memory_info.rss / (1024 ** 2)} MB")
    logging.info(f"GPU memory usage: {gpu_memory} MB")

logging.info("Starting distillation...")

# Load the pre-trained ViT feature extractor
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model_name_or_path = 'apple/mobilevit-xx-small'
feature_extractor = AutoImageProcessor.from_pretrained(model_name_or_path)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ]),
}

data_dir = '/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Sabina.Jangirova@mbzuai.ac.ae/fire_dataset'
train_dataset = datasets.ImageFolder(root=data_dir+'/train', transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root=data_dir+'/val', transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

labels = train_dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

model = AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=True,
    num_labels=len(train_dataset.classes),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts
class_weights /= class_weights.sum()  # Normalize the weights

# Move the class weights to the appropriate device
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

logging.info(model)
logging.info(summary(model))

# Training and validation loop
num_epochs = 100
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    running_val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            running_val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained('best_mobilevit_xxs')
        feature_extractor.save_pretrained('best_mobilevit_xxs')
        best_preds = all_preds
        best_labels = all_labels

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('losses_plot_mobilevit_xxs.png')
plt.close()

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(best_labels, best_preds)
class_report = classification_report(best_labels, best_preds, target_names=train_dataset.classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_mobilevit_xxs.png')
plt.close()

print('Classification Report:')
logging.info(class_report)
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
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

torch.cuda.empty_cache()


class DistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, student_features, teacher_features):
        # Logits distillation
        logits_loss = self.kl_div_loss(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # Feature distillation
        feature_loss = sum([self.mse_loss(sf, tf.detach()) for sf, tf in zip(student_features, teacher_features)])

        # Total loss
        return self.alpha * logits_loss + (1 - self.alpha) * feature_loss

class DistillationModel(nn.Module):
    def __init__(self, teacher, student):
        super(DistillationModel, self).__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, x):
        # Get teacher outputs and features
        teacher_outputs, teacher_features = self.get_features_and_output(self.teacher, x)

        # Get student outputs and features
        student_outputs, student_features = self.get_features_and_output(self.student, x)

        return student_outputs, teacher_outputs, student_features, teacher_features

    def get_features_and_output(self, model, x):
        features = []
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i in {2, 4, 6, 8}:  # Example: Extract features from certain layers
                features.append(x)
        x = model.norm(x)
        output = model.head(x.mean(dim=1))
        return output, features

# Setup logging
log_file = 'distill_vit16_fixed_37.log'
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
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ]),
}

# Load the datasets with ImageFolder
data_dir = '~/fire_detection/fire_detection/dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(image_datasets['train'].targets), y=image_datasets['train'].targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load the pre-trained ViT model
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
checkpoint_path = '/fsx/homes/Sabina.Jangirova@mbzuai.ac.ae/fire_detection/fire_detection/best_vit_model_weights_vit16_50_epochs.pth'
state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define student model
student_model_config = {
    'patch_size': 16,
    'depth': 4,        # Fewer transformer layers
    'num_heads': 4,    # Fewer attention heads
    'mlp_ratio': 6.0,
    'qkv_bias': True,
    'norm_layer': torch.nn.LayerNorm,
}
def create_custom_vit(config={}, pretrained=False, num_classes=12):
    model = timm.models.vision_transformer.VisionTransformer(
        img_size=224,
        patch_size=config['patch_size'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=config['qkv_bias'],
        norm_layer=config['norm_layer'],
        num_classes=num_classes
    )
    
    return model

student_model = create_custom_vit(config=student_model_config, pretrained=False, num_classes=num_classes)
student_model = student_model.to(device)

from torchsummary import summary

logging.info(summary(model))
logging.info(summary(student_model))

logging.info(model)
logging.info(student_model)

distillation_model = DistillationModel(model, student_model)
criterion = DistillationLoss(temperature=5.0, alpha=0.7)

train_losses = []
eval_losses = []
best_eval_loss = float('inf')
best_model_state = None

# Example training loop
num_epochs = 1000  # Set the number of epochs
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = 0.0
    distillation_model.train()

    for images, labels in dataloaders['train']:  # Assuming train_dataloader is defined
        optimizer.zero_grad()

        # Forward pass
        student_outputs, teacher_outputs, student_features, teacher_features = distillation_model(images)

        # Compute loss
        loss = criterion(student_outputs, teacher_outputs, student_features, teacher_features)
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Average train loss for the epoch
    train_loss /= len(dataloaders['train'])
    logging.info(f"Train Loss: {train_loss:.4f}")
    train_losses.append(train_loss)

    # Evaluation loop
    eval_loss = 0.0
    distillation_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloaders['val']:  # Assuming eval_dataloader is defined
            student_outputs, teacher_outputs, student_features, teacher_features = distillation_model(images)
            loss = criterion(student_outputs, teacher_outputs, student_features, teacher_features)
            eval_loss += loss.item()

            _, preds = torch.max(student_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Average evaluation loss for the epoch
    eval_loss /= len(dataloaders['val'])
    logging.info(f"Validation Loss: {eval_loss:.4f}")
    eval_losses.append(eval_loss)

    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    logging.info(f'Classification Report:\n{class_report}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')

    # Check if this is the best model so far
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_model_state = student_model.state_dict()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

# Save the best model based on evaluation loss
torch.save(best_model_state, "best_student_vit.pth")

# Plotting the training and evaluation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), eval_losses, label='Evaluation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.show()
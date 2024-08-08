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
from scipy.optimize import differential_evolution

torch.cuda.empty_cache()

# Define the modified ViT model with dual fire attention
class ViTWithDFA(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', config=None, num_classes=12):
        super(ViTWithDFA, self).__init__()
        self.vit = timm.models.vision_transformer.VisionTransformer(
            img_size=224,
            patch_size=16,
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            num_classes=0
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Sequential(
            nn.Linear(self.embed_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50)
        )

        self.conv_branch = nn.Sequential(
            nn.Conv2d(self.embed_dim, 64, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(64)
        )

        self.final_dense = nn.Sequential(
            nn.Linear(self.embed_dim + 64 + self.embed_dim, 150),
            nn.ReLU(),
            nn.BatchNorm1d(150),
            nn.Linear(150, num_classes)
        )

    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        for block in self.vit.blocks:
            x = block(x)

        x = self.vit.norm(x)
        cls_token_final = x[:, 0]
        cls_token_final_reshaped = cls_token_final.unsqueeze(-1).unsqueeze(-1)

        # Branch 1
        flat1 = self.global_avg_pool(cls_token_final_reshaped).flatten(1)
        x1 = self.dense1(flat1)

        # Branch 2
        x2 = self.conv_branch(cls_token_final_reshaped).flatten(1)

        # Branch 3
        x3 = self.global_avg_pool(cls_token_final_reshaped).flatten(1)

        # Concatenate branches
        concat = torch.cat([x1, x2], dim=1)
        concat = torch.cat([x3, concat], dim=1)

        # Final dense layers
        output = self.final_dense(concat)
        return output

# Define the Distiller class
class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        self.optimizer = optimizer
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        with torch.no_grad():
            teacher_predictions = self.teacher(x)
        student_predictions = self.student(x)
        student_loss = self.student_loss_fn(student_predictions, y)
        distillation_loss = self.distillation_loss_fn(
            torch.log_softmax(student_predictions / self.temperature, dim=1),
            torch.softmax(teacher_predictions / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        log_system_usage()
        logging.info(f"Train - Loss: {loss.item()}, Student Loss: {student_loss.item()}, Distillation Loss: {distillation_loss.item()}")
        return {"loss": loss.item(), "student_loss": student_loss.item(), "distillation_loss": distillation_loss.item()}

    def test_step(self, data):
        x, y = data
        with torch.no_grad():
            y_pred = self.student(x)
        student_loss = self.student_loss_fn(y_pred, y)
        logging.info(f"Test - Student Loss: {student_loss.item()}")
        return {"student_loss": student_loss.item()}

# Setup logging
log_file = 'distill_vit16_fixed_17.log'
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
    'depth': 6,        # Fewer transformer layers
    'num_heads': 6,    # Fewer attention heads
    'mlp_ratio': 6.0,
    'qkv_bias': True,
    'norm_layer': torch.nn.LayerNorm,
}

student_model = ViTWithDFA(config=student_model_config, num_classes=num_classes)
student_model = student_model.to(device)

optimizer = optim.Adam(student_model.parameters(), lr=0.00001)
distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')

distiller = Distiller(student=student_model, teacher=model)
distiller.compile(optimizer, criterion, distillation_loss_fn, alpha=0.1, temperature=5)

num_epochs = 500
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training phase
    distiller.train()
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        results = distiller.train_step((inputs, labels))
        running_loss += results["loss"]
    epoch_loss = running_loss / len(dataloaders['train'])
    train_losses.append(epoch_loss)
    logging.info(f"Train Loss: {epoch_loss:.4f}")
    
    # Validation phase
    distiller.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in dataloaders['val']:
        inputs, labels = inputs.to(device), labels.to(device)
        results = distiller.test_step((inputs, labels))
        running_loss += results["student_loss"]
        all_preds.extend(torch.argmax(distiller.student(inputs), dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloaders['val'])
    val_losses.append(epoch_loss)
    logging.info(f"Validation Loss: {epoch_loss:.4f}")

    # Save the best model
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        torch.save(student_model.state_dict(), '/tmp/best_student_vit16_model_fixed_17.pth')
        best_preds = all_preds
        best_labels = all_labels
    
    # Log classification report and confusion matrix for current epoch
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    logging.info(f'Classification Report:\n{class_report}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')

# Plot training and validation losses
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.savefig('losses_plot_distilled_fixed_17.png')
plt.close()

# Reload the best model weights
student_model.load_state_dict(torch.load('/tmp/best_student_vit16_model_fixed_17.pth'))

# Generate confusion matrix for the best model
distiller.eval()
all_preds = []
all_labels = []
for inputs, labels in dataloaders['val']:
    inputs, labels = inputs.to(device), labels.to(device)
    all_preds.extend(torch.argmax(distiller.student(inputs), dim=1).cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Log classification report and confusion matrix for the best model
class_report = classification_report(all_labels, all_preds, target_names=class_names)
conf_matrix = confusion_matrix(all_labels, all_preds)
logging.info(f'Best Model Classification Report:\n{class_report}')
logging.info(f'Best Model Confusion Matrix:\n{conf_matrix}')

# Plot confusion matrix for the best model
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Best Model Confusion Matrix')
plt.savefig('best_model_confusion_matrix_distilled_fixed_17.png')
plt.close()

# Save the trained student model
logging.info("Model saved successfully.")
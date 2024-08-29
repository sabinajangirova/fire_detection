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
from transformers import AutoImageProcessor, AutoModelForImageClassification

torch.cuda.empty_cache()

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
            teacher_predictions = self.teacher(x).logits  # Extract logits from the teacher
        student_predictions = self.student(x).logits  # Extract logits from the student
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
            y_pred = self.student(x).logits  # Extract logits from the student
        student_loss = self.student_loss_fn(y_pred, y)
        logging.info(f"Test - Student Loss: {student_loss.item()}")
        return {"student_loss": student_loss.item()}

# Setup logging
log_file = 'mobilevit_distill_1.log'
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
model_name_or_path = 'best_mobilevit_xxs'
feature_extractor = AutoImageProcessor.from_pretrained(model_name_or_path)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

data_dir = '/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Sabina.Jangirova@mbzuai.ac.ae/fire_dataset'
train_dataset = datasets.ImageFolder(root=data_dir+'/train', transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(root=data_dir+'/val', transform=data_transforms['val'])
class_names = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

labels = train_dataset.classes
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

student_model = AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(train_dataset.classes),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_model = student_model.to(device)
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts
class_weights /= class_weights.sum()  # Normalize the weights

# Move the class weights to the appropriate device
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load the pre-trained ViT model
teacher_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(train_dataset.classes))
checkpoint_path = '/fsx/homes/Sabina.Jangirova@mbzuai.ac.ae/fire_detection/fire_detection/best_vit_model_weights_vit16_50_epochs.pth'
state_dict = torch.load(checkpoint_path, map_location='cpu')
teacher_model.load_state_dict(state_dict)
model = teacher_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)

from torchsummary import summary

logging.info(summary(teacher_model))
logging.info(summary(student_model))

logging.info(teacher_model)
logging.info(student_model)

optimizer = optim.Adam(student_model.parameters(), lr=0.0001)
distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')

distiller = Distiller(student=student_model, teacher=model)
distiller.compile(optimizer, criterion, distillation_loss_fn, alpha=0.1, temperature=5)

num_epochs = 50
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training phase
    distiller.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        results = distiller.train_step((inputs, labels))
        running_loss += results["loss"]
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    logging.info(f"Train Loss: {epoch_loss:.4f}")
    
    # Validation phase
    distiller.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        results = distiller.test_step((inputs, labels))
        running_loss += results["student_loss"]
        all_preds.extend(torch.argmax(distiller.student(inputs), dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(val_loader)
    val_losses.append(epoch_loss)
    logging.info(f"Validation Loss: {epoch_loss:.4f}")

    # Save the best model
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        student_model.save_pretrained('distilled_mobilevit_xxs_1')
        feature_extractor.save_pretrained('distilled_mobilevit_xxs_1')
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
plt.savefig('distilled_mobilevit_xxs_1_losses.png')
plt.close()

# Reload the best model weights
student_model.load_state_dict(torch.load('distilled_mobilevit_xxs_1'))

# Generate confusion matrix for the best model
distiller.eval()
all_preds = []
all_labels = []
for inputs, labels in val_loader:
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
plt.savefig('distilled_mobilevit_xxs_1_confusion.png')
plt.close()

# Save the trained student model
logging.info("Model saved successfully.")
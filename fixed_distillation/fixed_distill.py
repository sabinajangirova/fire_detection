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

# Function to create a student model based on the modified configuration
def create_student_vit(config, num_classes):
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

student_model = create_student_vit(student_model_config, num_classes)
student_model = student_model.to(device)

optimizer = optim.Adam(student_model.parameters(), lr=0.00001)
distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')

distiller = Distiller(student=student_model, teacher=model)
distiller.compile(optimizer, criterion, distillation_loss_fn, alpha=0.1, temperature=5)

def prune_model(model, neuron_mask):
    for i, mask in enumerate(neuron_mask):
        layer = model.blocks[i]
        if hasattr(layer, 'mlp'):
            pruned_out_features = int(sum(mask))
            in_features = layer.mlp.fc1.weight.shape[1]
            fc1 = nn.Linear(in_features, pruned_out_features, bias=layer.mlp.fc1.bias is not None).to(device)
            fc2 = nn.Linear(pruned_out_features, layer.mlp.fc2.weight.shape[0], bias=layer.mlp.fc2.bias is not None).to(device)
            with torch.no_grad():
                fc1.weight.copy_(layer.mlp.fc1.weight[:, mask])
                fc1.bias.copy_(layer.mlp.fc1.bias[mask] if layer.mlp.fc1.bias is not None else None)
                fc2.weight.copy_(layer.mlp.fc2.weight[mask, :])
                fc2.bias.copy_(layer.mlp.fc2.bias if layer.mlp.fc2.bias is not None else None)
            layer.mlp.fc1 = fc1
            layer.mlp.fc2 = fc2
            layer.mlp.fc1.out_features = pruned_out_features

def evaluate_student(neuron_mask, model, X_train, y_train):
    prune_model(model, neuron_mask)
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        accuracy = (outputs.argmax(dim=1) == y_train).float().mean().item()
    return -accuracy  # We minimize the negative accuracy

def meta_heuristic_pruning(model, X_train, y_train):
    num_neurons = [layer.mlp.fc1.out_features for layer in model.blocks if hasattr(layer, 'mlp')]
    bounds = [(0, 1)] * sum(num_neurons)

    def prune_model(weights):
        mask = np.split(weights > 0.5, np.cumsum(num_neurons[:-1]))
        return mask

    result = differential_evolution(
        lambda w: evaluate_student(prune_model(w), model, X_train, y_train),
        bounds, strategy='best1bin', maxiter=100, popsize=15
    )

    best_mask = prune_model(result.x)
    return best_mask

# Prune the student model
logging.info("Pruning the student model...")
X_train, y_train = next(iter(dataloaders['train']))
X_train, y_train = X_train.to(device), y_train.to(device)
best_mask = meta_heuristic_pruning(student_model, X_train, y_train)
logging.info(f"Best mask for neurons: {best_mask}")
prune_model(student_model, best_mask)

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
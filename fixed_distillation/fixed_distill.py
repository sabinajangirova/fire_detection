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
torch.cuda.empty_cache()
class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        """
        Configure the distiller.

        Args:
            optimizer: PyTorch optimizer for the student weights
            student_loss_fn: Loss function of difference between student predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions. Larger temperature gives softer distributions.
        """
        self.optimizer = optimizer
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        with torch.no_grad():
            teacher_predictions = self.teacher(x)

        # Forward pass of student
        student_predictions = self.student(x)

        # Compute losses
        student_loss = self.student_loss_fn(student_predictions, y)
        distillation_loss = self.distillation_loss_fn(
            torch.log_softmax(student_predictions / self.temperature, dim=1),
            torch.softmax(teacher_predictions / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log system usage
        log_system_usage()

        logging.info(f"Train - Loss: {loss.item()}, Student Loss: {student_loss.item()}, Distillation Loss: {distillation_loss.item()}")

        return {
            "loss": loss.item(),
            "student_loss": student_loss.item(),
            "distillation_loss": distillation_loss.item()
        }

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        with torch.no_grad():
            y_pred = self.student(x)

        # Calculate the loss
        student_loss = self.student_loss_fn(y_pred, y)

        logging.info(f"Test - Student Loss: {student_loss.item()}")

        return {
            "student_loss": student_loss.item()
        }

# Setup logging
log_file = 'distill_vit16_fixed_17.log'
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s %(message)s')

def log_system_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    gpu_memory = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0
    logging.info(f"CPU usage: {cpu_usage}%")
    logging.info(f"Memory usage: {memory_info.rss / (1024 ** 2)} MB")
    logging.info(f"GPU memory usage: {gpu_memory} MB")

logging.info("Starting distillation...")

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

# model.head = nn.Identity()

# student_model_config = {
#     'patch_size': 16,
#     'embed_dim': 192,  # Smaller embedding dimension
#     'depth': 8,        # Fewer transformer layers
#     'num_heads': 6,    # Fewer attention heads
#     'mlp_ratio': 4.0,
#     'qkv_bias': True,
#     'norm_layer': torch.nn.LayerNorm,
# }

student_model_config = {
    'patch_size': 16,
    'depth': 6,        # Fewer transformer layers
    'num_heads': 6,    # Fewer attention heads
    'mlp_ratio': 6.0,
    'qkv_bias': True,
    'norm_layer': torch.nn.LayerNorm,
}

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, patch_size, embed_dim, num_shifts=4):
        super(ShiftedPatchTokenization, self).__init__()
        self.patch_size = patch_size
        self.num_shifts = num_shifts
        self.patch_embed = nn.Linear(patch_size * patch_size * 3 * (num_shifts + 1), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        shifts = [
            (0, 0),  # original
            (-self.patch_size // 2, -self.patch_size // 2),  # top-left
            (-self.patch_size // 2, self.patch_size // 2),   # top-right
            (self.patch_size // 2, -self.patch_size // 2),   # bottom-left
            (self.patch_size // 2, self.patch_size // 2)     # bottom-right
        ]
        patches = []
        for dx, dy in shifts:
            shifted_x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
            patch = shifted_x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            patch = patch.contiguous().view(B, C, -1, self.patch_size * self.patch_size).permute(0, 2, 3, 1)
            patch = patch.contiguous().view(B, -1, self.patch_size * self.patch_size * C)
            patches.append(patch)
        tokens = torch.cat(patches, dim=-1)
        tokens = self.patch_embed(tokens)
        tokens = self.norm(tokens)
        return tokens

class LocalitySelfAttention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0., temperature=1.0):
        super(LocalitySelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * temperature)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = attn / self.temperature
        attn = attn.masked_fill(torch.eye(N, device=attn.device).bool(), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ImprovedViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=12, embed_dim=768, depth=6, num_heads=6, mlp_ratio=6., qkv_bias=False, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super(ImprovedViT, self).__init__()
        self.num_classes = num_classes
        self.patch_embed = ShiftedPatchTokenization(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                norm_layer(embed_dim),
                LocalitySelfAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate),
                nn.Dropout(drop_rate),
                norm_layer(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(drop_rate)
                )
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

student_model = ImprovedViT()

# Function to create a student model based on the modified configuration
#def create_student_vit(config, num_classes):
#    model = timm.models.vision_transformer.VisionTransformer(
#        img_size=224,
#        patch_size=config['patch_size'],
#        depth=config['depth'],
#        num_heads=config['num_heads'],
#        mlp_ratio=config['mlp_ratio'],
#        qkv_bias=config['qkv_bias'],
#        norm_layer=config['norm_layer'],
#        num_classes=num_classes
#    )
#    
#    return model
    
#student_model = create_student_vit(student_model_config, num_classes)
#student_model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=num_classes)
# Verify the model
logging.info(student_model)

#student_model_scratch = create_student_vit(student_model_config, num_classes)

from torchsummary import summary

logging.info(summary(model))
logging.info(summary(student_model))

optimizer = optim.Adam(student_model.parameters(), lr=0.00001)
distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')

distiller = Distiller(student=student_model, teacher=model)
distiller.compile(optimizer, criterion, distillation_loss_fn, alpha=0.1, temperature=5)

student_model.to(device)

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

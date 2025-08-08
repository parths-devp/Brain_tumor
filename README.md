
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import transforms 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

data = r"C:\Users\parth\Downloads\archive (5)\brain_tumor_dataset"
yes_dir = os.path.join(data, "yes")
no_dir = os.path.join(data, "no")
yes_images = [f for f in os.listdir(yes_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
no_images = [f for f in os.listdir(no_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
            return image, torch.tensor(label, dtype=torch.float32)

yes_paths = [os.path.join(yes_dir, img) for img in yes_images]
no_paths = [os.path.join(no_dir, img) for img in no_images]
image_paths = yes_paths + no_paths
labels = [1] * len(yes_paths) + [0] * len(no_paths)

combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

x_temp, x_test, y_temp, y_test = train_test_split(image_paths, labels, test_size=0.15, stratify=labels, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

train_dataset = BrainTumorDataset(x_train, y_train, transform)
val_dataset = BrainTumorDataset(x_val, y_val, transform)
test_dataset = BrainTumorDataset(x_test, y_test, transform)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class CNNBrainTumorClassifier(nn.Module):
    def __init__(self):
        super(CNNBrainTumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = CNNBrainTumorClassifier().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

def binary_accuracy(outputs, labels):
    preds = (outputs > 0.5).float()
    correct = (preds == labels).float()
    return correct.sum() / len(correct)

epochs = 5
best_val_acc = 0.0
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += binary_accuracy(outputs, labels).item()
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += binary_accuracy(outputs, labels).item()
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    scheduler.step(avg_val_loss)
    print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), 'best_brain_tumor_model.pth')

model.eval()
test_loss = 0.0
test_acc = 0.0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_acc += binary_accuracy(outputs, labels).item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
avg_test_acc = test_acc / len(test_loader)
print(f"\nTest Results:")
print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {avg_test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['No Tumor', 'Tumor']))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

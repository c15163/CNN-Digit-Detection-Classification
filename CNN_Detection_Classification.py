import os
import cv2
import numpy as np
import mat73
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

train_path = './train'
test_path = './test'
mat_train_path = './train/digitStruct.mat'
npz_train_path = './train/train_dataset_with_non_digit.npz'
mat_test_path = './test/digitStruct.mat'
npz_test_path = './test/test_dataset_with_non_digit.npz'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(npz_train_path):  # If npz training file exists, load the file.
    train_cached = np.load(npz_train_path, allow_pickle=True)
    print('train data is already loaded!')
    crop_images = train_cached['images']
    train_labels = train_cached['labels']
    train_data_info = train_cached['info'].tolist()
else:
    print('Lets make train data!')
    data_dict = mat73.loadmat(mat_train_path)
    all_train_images_info = data_dict['digitStruct']
    image_names = all_train_images_info['name']
    bbox_infos = all_train_images_info['bbox']


    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    train_crop_images = []
    train_labels = []
    train_data_info = []

    for i in range(len(image_names)):
        image_name = image_names[i]
        image_path = os.path.join(train_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        hi, wi = image.shape[:2]
        bbox = bbox_infos[i]
        lefts = bbox['left']
        tops = bbox['top']
        widths = bbox['width']
        heights = bbox['height']
        labels = bbox['label']

        if not isinstance(lefts, list):
            lefts = [lefts]
            tops = [tops]
            widths = [widths]
            heights = [heights]
            labels = [labels]

        boxes = []
        new_labels = []

        for x, y, w, h, label in zip(lefts, tops, widths, heights, labels):
            x, y, w, h = int(x), int(y), int(w), int(h)
            boxes.append((x, y, w, h))
            crop = image[y:y + h, x:x + w]
            if crop is not None and crop.size > 0:
                resized = cv2.resize(crop, (32, 32))
                train_crop_images.append(resized)
                lab = 0 if int(label) == 10 else int(label)
                train_labels.append(lab)
                new_labels.append(lab)

        # non-digit box at (0, 0) with same size as first box. In the original dataset, just 10 digits are in the data. Non-digit class is necessary.
        if boxes and i % 4 == 0:
            w0, h0 = boxes[0][2], boxes[0][3]
            non_box = (0, 0, w0, h0)
            for box in boxes:
                if compute_iou(non_box,box) >= 0.1:  # if the non_box is overlapped with the previous boxes more than 10%, eliminate it.
                    break
            else:
                crop = image[0:h0, 0:w0]
                if crop is not None:
                    resized = cv2.resize(crop, (32, 32))
                    train_crop_images.append(resized)
                    train_labels.append(10)
                    boxes.append((0, 0, w0, h0))
                    new_labels.append(10)  # To save in processed_info

        train_data_info.append({
            "image_name": image_name,
            "left": [b[0] for b in boxes],
            "top": [b[1] for b in boxes],
            "width": [b[2] for b in boxes],
            "height": [b[3] for b in boxes],
            "label": new_labels
        })

    # Store as a cache
    np.savez_compressed(npz_train_path, images=np.array(train_crop_images, dtype=np.uint8),labels=np.array(train_labels), info=np.array(train_data_info, dtype=object))

if os.path.exists(npz_test_path):  # If npz test file exists, load the file.
    test_cached = np.load(npz_test_path, allow_pickle=True)
    print('test data is already loaded!')
    crop_images = test_cached['images']
    test_labels = test_cached['labels']
    test_data_info = test_cached['info'].tolist()
else:
    print('Lets make test data!')
    data_dict = mat73.loadmat(mat_test_path)
    all_test_images_info = data_dict['digitStruct']
    image_names = all_test_images_info['name']
    bbox_infos = all_test_images_info['bbox']


    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


    test_crop_images = []
    test_labels = []
    test_data_info = []  # For printout

    for i in range(len(image_names)):
        image_name = image_names[i]
        image_path = os.path.join(test_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        hi, wi = image.shape[:2]
        bbox = bbox_infos[i]
        lefts = bbox['left']
        tops = bbox['top']
        widths = bbox['width']
        heights = bbox['height']
        labels = bbox['label']

        if not isinstance(lefts, list):
            lefts = [lefts]
            tops = [tops]
            widths = [widths]
            heights = [heights]
            labels = [labels]

        boxes = []
        new_labels = []

        for x, y, w, h, label in zip(lefts, tops, widths, heights, labels):
            x, y, w, h = int(x), int(y), int(w), int(h)
            boxes.append((x, y, w, h))
            crop = image[y:y + h, x:x + w]
            if crop is not None and crop.size > 0:
                resized = cv2.resize(crop, (32, 32))
                test_crop_images.append(resized)
                lab = 0 if int(label) == 10 else int(label)
                test_labels.append(lab)
                new_labels.append(lab)

        # non-digit box at (0, 0) with same size as first box. In the original dataset, just 10 digits are in the data. Non-digit class is necessary.
        if boxes and i % 4 == 0:
            w0, h0 = boxes[0][2], boxes[0][3]
            non_box = (0, 0, w0, h0)
            for box in boxes:
                if calculate_iou(non_box,box) >= 0.1:  # if the non_box is overlapped with the previous boxes more than 10%, eliminate it.
                    break
            else:
                crop = image[0:h0, 0:w0]
                if crop is not None:
                    resized = cv2.resize(crop, (32, 32))
                    test_crop_images.append(resized)
                    test_labels.append(10)
                    boxes.append((0, 0, w0, h0))
                    new_labels.append(10)  # To save in processed_info

        test_data_info.append({
            "image_name": image_name,
            "left": [b[0] for b in boxes],
            "top": [b[1] for b in boxes],
            "width": [b[2] for b in boxes],
            "height": [b[3] for b in boxes],
            "label": new_labels
        })

    # Store as a cache
    np.savez_compressed(npz_test_path, images=np.array(test_crop_images, dtype=np.uint8), labels=np.array(test_labels),info=np.array(test_data_info, dtype=object))

npz_data = np.load('./train/train_dataset_with_non_digit.npz', allow_pickle=True)
images = npz_data['images']
labels = npz_data['labels']

# Make 20% of train dataset as a validation dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Making a Dataset (From https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = image.astype(np.uint8)
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = MyDataset(X_train, y_train, transform=transform)
val_dataset = MyDataset(X_val, y_val, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Total Train Label Distribution:", Counter(train_labels))
print("Actual Train Label Distribution:", Counter(y_train))
print("Validation Label Distribution:", Counter(y_val))
print("Test Label Distribution:", Counter(test_labels))

# Simplified VGG model
class Simple_VGG(nn.Module):
    def __init__(self, num_classes=11, in_channels=3):
        super(Simple_VGG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 64 x 16 X 16
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 128 x 8 X 8
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 256 x 4 X 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 512 x 2 X 2
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 512 x 1 X 1
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),   # if block 4 or 5 are used, the dimension should be changed. In this case, block 3 is used.
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Pretrained VGG code up to block 3.
class PretrainedVGG(nn.Module):
    def __init__(self, num_classes=11):
        super(PretrainedVGG, self).__init__()
        vgg = models.vgg16(pretrained=True)
        # Freeze all feature extractor layers and use up to block 3.
        for param in vgg.features.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(vgg.features.children())[:17])  # block 1 has 5 layers, B2 has 5 layers, B3 has 7 layers. (4 and 5 has 7 respectively.)
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# This is my own architecture.
class MyCNN(nn.Module):
    def __init__(self, num_classes=11, in_channels=3):
        super(MyCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 16 x 16 X 16
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 32 x 8 X 8
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # output size = 64 x 4 X 4
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),  # To avoid overfitting.
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),  # To avoid overfitting.
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Focal loss is used due to the class imbalance.
def focal_loss(logits, targets, gamma=2.0, alpha=None):
    if alpha is not None:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=alpha)
    else:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = ((1 - pt) ** gamma) * ce_loss
    return loss.mean()

num_class = 11
label_distribution = Counter(train_labels)
counts = torch.tensor([label_distribution[i] for i in range(num_class)], dtype=torch.float)
total = counts.sum()
alpha = total / (counts * len(counts))
alpha = alpha / alpha.sum()
alpha = alpha.to(device)  # to(device)

def train_and_evaluate(model_class, model_name, lr=0.0001, use_focal_loss=False, alpha=None, num_classes=11, epochs=20, waiting=3):
    model = model_class(num_classes=num_classes).to(device)
    if use_focal_loss:
        criterion = lambda logits, targets: focal_loss(logits, targets, gamma=2.0, alpha=alpha)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    count = 0  # how many times the model reaches the plateau.
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print(f"Training {model_name} with lr={lr}")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        # Early stopping criteria
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            count = 0
            torch.save(model.state_dict(), f'{model_name}_best_model_{lr}.pth')
        else:
            count += 1
            print(f"Does not improve for {count} epochs.")
            if count >= waiting:
                print(f'Early stopping at epoch {epoch+1}')
                #break  # If all losses or accuracies are necessary, please comment out break.
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%")

    runtime = time.time() - start_time
    print(f"{model_name} training took {runtime:.2f} seconds.")

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Loss per Epoch')
    plt.savefig(f'[{model_name}]loss_plot_{lr}_1.png')
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{model_name} Accuracy per Epoch')
    plt.savefig(f'[{model_name}]accuracy_plot_{lr}_1.png')
    plt.close()

    # Test Evaluation
    test_npz = np.load('./test/test_dataset_with_non_digit.npz', allow_pickle=True)
    test_images = test_npz['images']
    test_labels = test_npz['labels']
    test_dataset = MyDataset(test_images, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.load_state_dict(torch.load(f'{model_name}_best_model_{lr}.pth'))
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)
    test_accuracy = 100 * test_correct / test_total
    print(f"{model_name} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


def run_my_cnn(lr=0.0005, waiting=3):
    train_and_evaluate(MyCNN, 'MyCNN', lr=lr, use_focal_loss=True, alpha=alpha, waiting=waiting)

def run_simple_vgg(lr=0.0001, waiting=3):
    train_and_evaluate(Simple_VGG, 'Simple_VGG', lr=lr, use_focal_loss=False, waiting=waiting)

def run_pretrained_vgg(lr=0.0001, waiting=3):
    train_and_evaluate(PretrainedVGG, 'PretrainedVGG', lr=lr, use_focal_loss=False, waiting=waiting)

########################### Dectection Code #############################
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def non_max_suppression(boxes, iou_threshold=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    picked_boxes = []
    while boxes:
        current = boxes.pop(0)
        picked_boxes.append(current)
        boxes = [b for b in boxes if calculate_iou(current, b) < iou_threshold]
    return picked_boxes

def detect_digits_mser_pyramid(image, scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], contrast_threshold = 10):
    original = image.copy()
    all_boxes = []

    for scale in scales:
        img = cv2.resize(original, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        mser = cv2.MSER_create(delta=5, min_area=100, max_area=5000)
        regions, _ = mser.detectRegions(gray)
        #edges = cv2.Canny(gray, 100, 200)

        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w/scale < 10 or h/scale < 10:
                continue
            ratio = w / h
            if ratio < 0.2 or ratio > 2.5:  # Too narrow or too wide boxes are eliminated.
                continue
            patch = gray[y:y + h, x:x + w]
            if patch.std() < contrast_threshold:  # Contrast filtering
                continue

            # restore to the original coordinates
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            all_boxes.append((x_orig, y_orig, w_orig, h_orig))

    return all_boxes
    #non_max_suppression(all_boxes, iou_threshold=0.2)

def detection_and_classification(train_path, output_path, model, best_file_path):
    model.load_state_dict(torch.load(best_file_path))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for file in os.listdir(train_path):
        img_path = os.path.join(train_path, file)
        image = cv2.imread(img_path)
        img_for_detected = image.copy()
        img_for_classification = image.copy()

        boxes = detect_digits_mser_pyramid(image)
        boxes = non_max_suppression(boxes, iou_threshold=0.1)  # iou is 10 %.
        crop_images = []
        final_boxes = []
        for (x, y, w, h) in boxes:
            crop = image[y:y+h, x:x+w]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            resized = cv2.resize(crop, (32, 32))
            crop_images.append(resized)
            final_boxes.append((x, y, w, h))
            # Save the detected image
            cv2.rectangle(img_for_detected, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Detection-only save
        base_name = os.path.splitext(file)[0]
        cv2.imwrite(os.path.join(output_path, f'{base_name}_detected.png'), img_for_detected)
        # Classification
        for crop, (x, y, w, h) in zip(crop_images, final_boxes):
            input_tensor = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                max_prob, pred = torch.max(probs, dim=1)
                #pred = torch.argmax(output, dim=1).item()
                if max_prob.item() < 0.5 or pred.item()==10:  # If the maximum prob is less than 50%, it will be classified to 'Non-digit'
                    continue
            cv2.rectangle(img_for_classification, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_for_classification, str(pred.item()), (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

        # Detection + Classification 저장
        cv2.imwrite(os.path.join(output_path, f'{base_name}_detection_and_classification.png'), img_for_classification)


lr = 0.001  # MyCNN has the best result when lr = 0.0005, Simple VGG has the best result when lr = 0.0001
waiting = 3  # Early stop criteria. Just wait three epoch when the model reached plateau.
# Train Simplified VGG, Pretrained VGG, and MyCNN models
#run_my_cnn(lr=lr, waiting=waiting)
#run_simple_vgg(lr=lr, waiting=waiting)
#run_pretrained_vgg(lr=lr, waiting=waiting)
lr = 0.001
run_simple_vgg(lr=lr, waiting=waiting)
lr = 0.0005
run_simple_vgg(lr=lr, waiting=waiting)

train_path = './house_numbers'
output_path = './output_images'
model = MyCNN(num_classes=11).to(device)   # If VGG is necessary, it can be changed to VGG.
best_file_path = './MyCNN_best_model_0.0005.pth'

#detection_and_classification(train_path, output_path, model, best_file_path)


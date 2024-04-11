import os
import re
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,random_split
from sklearn.model_selection import StratifiedKFold
from albumentations import (
    Compose,
    HorizontalFlip,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    Transpose,
    Resize,
    CenterCrop,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
import timm

class EarlyStopping:
    def __init__(self, patience, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop
    
class GetDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.file_names = df["image_id"].values
        self.labels = df["label"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f"{self.path}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    
def get_transforms(data):
    if data == "train":
        return Compose([
        RandomResizedCrop(384,384),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2,sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
        ])
    elif data == "valid":
        return Compose(
            [
                Resize(384, 384),
                CenterCrop(384, 384),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    
def get_train():
    #load dataset
    train = pd.read_csv("./Dataset/train.csv")
    sns.countplot(x='label', data=train)
    plt.savefig("./task/count.png")

    #sample data
    label_counts = train["label"].value_counts()
    sample_ratios = {
       label: 0.2 for label in label_counts.index  
    }
    sampled_data = []
    for label, count in label_counts.items():
       label_data = train[train["label"] == label]
       num_samples = int(count * sample_ratios[label])
       sampled_data.append(label_data.sample(n=num_samples, replace=False))

    train_sampled = pd.concat(sampled_data)

    #split folder
    train_copy = train_sampled.copy()
    train_copy.reset_index(drop=True, inplace=True)
    train_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(train_fold.split(train_copy, train_copy["label"])):
        train_copy.loc[val_index, "fold"] = int(n)
    train_copy["fold"] = train_copy["fold"].astype(int)

    return train_copy

def train_loop(train_copy,fold):
    #load model
    resnext = models.resnext50_32x4d(pretrained=True)
    resnext.fc = nn.Linear(resnext.fc.in_features, 5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnext.parameters(), lr=0.0001)

    print(f"========== fold: {fold} training ==========")

    train_idx = train_copy[train_copy["fold"] != fold].index
    val_idx = train_copy[train_copy["fold"] == fold].index

    train_folds = train_copy.loc[train_idx].reset_index(drop=True)
    valid_folds = train_copy.loc[val_idx].reset_index(drop=True)

    train_dataset = GetDataset(train_folds,path = "./Dataset/train_images", transform=get_transforms(data = "train"))
    valid_dataset = GetDataset(valid_folds,path = "./Dataset/train_images", transform=get_transforms(data = "valid"))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    epochs = 10
    early_stopping = EarlyStopping(patience=3, verbose=True)

    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        for images, labels in tqdm(train_loader):
          optimizer.zero_grad()

          #mixup
          indices = torch.randperm(images.size(0))
          images_s, labels_s = images[indices], labels[indices]
          lam = np.random.beta(0.5, 0.5)
          images_mix = lam * images + (1 - lam) * images_s

          outputs = resnext(images_mix)
          loss = criterion(outputs, labels) * lam + criterion(outputs, labels_s) * (1 - lam)
          loss.backward()
          optimizer.step()
    
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = resnext(images)
                loss = criterion(outputs, labels)
                loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        loss /= len(valid_loader)
        val_losses.append(loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            print(f"Epoch {epoch+1} - Save Best Score: {best_acc:.4f} Model")
            torch.save(
                {"model": resnext.state_dict()}, "./task/"+ f"{'resnext'}_fold{fold}_best.pth"
            )
        if early_stopping(loss):
            print("Early stopping")
            break
    
    return val_losses, val_accuracies

def validate_model(train_copy,path):

    valid_dataset = GetDataset(train_copy,path = "./Dataset/train_images", transform=get_transforms(data = "valid"))
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(path))
    model.eval()
  
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
               outputs = model(images)
               loss = criterion(outputs, labels)
               loss += loss.item()
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()

    loss /= len(valid_loader)
    accuracy = correct / total

    return loss, accuracy

def extract_numbers(string):
    numbers = re.findall(r'\d+\.\d+', string)
    return [float(num) for num in numbers]

def plot_accloss():
  df = pd.read_csv('./task/validation_results2.csv')
  df['val_losses'] = df['val_losses'].apply(extract_numbers)
  df['val_accuracies'] = df['val_accuracies'].apply(extract_numbers)
  fig, axes = plt.subplots(2, 1, figsize=(10, 8))

  loss_ax = axes[0]
  loss_ax.grid(True)
  loss_data = []
  for fold in range(5):
      fold_data = df[df['fold'] == fold]
      loss_data.extend(fold_data.iloc[0]['val_losses'])
      if fold != 5:
          loss_ax.axvline(x=len(loss_data), color='gray', linestyle='--')
          loss_ax.text(len(loss_data) - len(fold_data.iloc[0]['val_losses']) / 2, max(loss_data), f"Fold {fold}",
                      ha='center', va='bottom')

  epoch_counts_loss = np.arange(1, len(loss_data) + 1)
  loss_ax.plot(epoch_counts_loss, loss_data)
  loss_ax.set_title('Loss')
  loss_ax.set_xlabel('Epochs')
  loss_ax.set_ylabel('Loss')

  accuracy_ax = axes[1]
  accuracy_ax.grid(True) 
  accuracy_data = []
  for fold in range(5):
      fold_data = df[df['fold'] == fold]
      accuracy_data.extend(fold_data.iloc[0]['val_accuracies'])
      if fold != 5:
          accuracy_ax.axvline(x=len(accuracy_data), color='gray', linestyle='--')
          accuracy_ax.text(len(accuracy_data) - len(fold_data.iloc[0]['val_accuracies']) / 2, 0.9,
                          f"Fold {fold}", ha='center', va='bottom')

  epoch_counts_accuracy = np.arange(1, len(accuracy_data) + 1)
  accuracy_ax.plot(epoch_counts_accuracy, accuracy_data)
  accuracy_ax.set_title('Accuracy')
  accuracy_ax.set_xlabel('Epochs')
  accuracy_ax.set_ylabel('Accuracy')
  plt.tight_layout()
  plt.show()
  plt.savefig("./task/accloss.png")

def fill_missing_epochs(df, df_ref):
    for fold in range(5):
        fold_data = df[df['fold'] == fold]
        fold_data_ref = df_ref[df_ref['fold'] == fold]
        num_epochs = len(fold_data['val_losses'].values[0])
        num_epochs_ref = len(fold_data_ref['val_losses'].values[0])
        if num_epochs < num_epochs_ref:
            last_epoch_loss = fold_data.iloc[-1]['val_losses'][-1]
            last_epoch_accuracy = fold_data.iloc[-1]['val_accuracies'][-1]
            for i in range(num_epochs, num_epochs_ref):
                fold_data.iloc[0]['val_losses'].append(last_epoch_loss)
                fold_data.iloc[0]['val_accuracies'].append(last_epoch_accuracy)
    return df

def plot_2accloss(path1,path2,label1,label2,savepath):
  df = pd.read_csv(path1)
  df2 = pd.read_csv(path2)
  df['val_losses'] = df['val_losses'].apply(extract_numbers)
  df['val_accuracies'] = df['val_accuracies'].apply(extract_numbers)
  df2['val_losses'] = df2['val_losses'].apply(extract_numbers)
  df2['val_accuracies'] = df2['val_accuracies'].apply(extract_numbers)

  df = fill_missing_epochs(df, df2)
  df2 = fill_missing_epochs(df2, df)

  fig, axes = plt.subplots(2, 1, figsize=(10, 8))

  loss_ax = axes[0]
  loss_ax.grid(True)
  loss_data = []
  for fold in range(5):
      fold_data = df[df['fold'] == fold]
      loss_data.extend(fold_data.iloc[0]['val_losses'])
      if fold != 5:
          loss_ax.axvline(x=len(loss_data), color='gray', linestyle='--')
          loss_ax.text(len(loss_data) - len(fold_data.iloc[0]['val_losses']) / 2, max(loss_data), f"Fold {fold}",
                      ha='center', va='bottom')

  epoch_counts_loss = np.arange(1, len(loss_data) + 1)
  loss_ax.plot(epoch_counts_loss, loss_data, label=label1)

  loss_data2 = []
  for fold in range(5):
     fold_data = df2[df2['fold'] == fold]
     loss_data2.extend(fold_data.iloc[0]['val_losses'])

  epoch_counts_loss2 = np.arange(1, len(loss_data2) + 1)
  loss_ax.plot(epoch_counts_loss2, loss_data2, label= label2)
  loss_ax.set_title('Loss')
  loss_ax.set_xlabel('Epochs')
  loss_ax.set_ylabel('Loss')
  loss_ax.legend()

  accuracy_ax = axes[1]
  accuracy_ax.grid(True) 
  accuracy_data = []
  for fold in range(5):
      fold_data = df[df['fold'] == fold]
      accuracy_data.extend(fold_data.iloc[0]['val_accuracies'])
      if fold != 5:
          accuracy_ax.axvline(x=len(accuracy_data), color='gray', linestyle='--')
          accuracy_ax.text(len(accuracy_data) - len(fold_data.iloc[0]['val_accuracies']) / 2, 0.9,
                          f"Fold {fold}", ha='center', va='bottom')

  epoch_counts_accuracy = np.arange(1, len(accuracy_data) + 1)
  accuracy_ax.plot(epoch_counts_accuracy, accuracy_data, label=label1)

  accuracy_data2 = []
  for fold in range(5):
      fold_data = df2[df2['fold'] == fold]
      accuracy_data2.extend(fold_data.iloc[0]['val_accuracies'])

  epoch_counts_accuracy2 = np.arange(1, len(accuracy_data2) + 1)
  accuracy_ax.plot(epoch_counts_accuracy2, accuracy_data2, label=label2)
  accuracy_ax.set_title('Accuracy')
  accuracy_ax.set_xlabel('Epochs')
  accuracy_ax.set_ylabel('Accuracy')
  accuracy_ax.legend()
  plt.tight_layout()
  plt.show()
  plt.savefig("savepath")

class ResNetWithCAM(nn.Module):
    def __init__(self, resnet):
        super(ResNetWithCAM, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = resnet.fc
        self.resnet = resnet

    def forward(self, x):
        x = self.features(x)
        feature_map = x.detach()  # save feature map for cam
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, feature_map
    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_spm(input,target,model):
    imgsize = (384, 384)
    bs = input.size(0)
    with torch.no_grad():
        output,fms = model(input)
        clsw = model.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea), dim=-1)
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i,target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i,target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()


    return outmaps,clslogit


def snapmix(input, target, alpha, model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if True:
        wfmaps,_ = get_spm(input, target, model)
        bs = input.size(0)
        lam = np.random.beta(alpha, alpha)
        lam1 = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(bs)
        wfmaps_b = wfmaps[rand_index,:,:]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

        area = (bby2-bby1)*(bbx2-bbx1)
        area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

        if  area1 > 0 and  area>0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(int(bbx2-bbx1), int(bby2-bby1)), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
            lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1-lam

    return input,target,target_b,lam_a,lam_b

def train_snapmix(train_copy,fold):
    #load model
    resnext = models.resnext50_32x4d(pretrained=True)
    resnext.fc = nn.Linear(resnext.fc.in_features, 5)
    resnext_cam = ResNetWithCAM(resnext)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnext.parameters(), lr=0.0001)

    print(f"========== fold: {fold} training ==========")

    train_idx = train_copy[train_copy["fold"] != fold].index
    val_idx = train_copy[train_copy["fold"] == fold].index

    train_folds = train_copy.loc[train_idx].reset_index(drop=True)
    valid_folds = train_copy.loc[val_idx].reset_index(drop=True)

    train_dataset = GetDataset(train_folds,path = "./Dataset/train_images", transform=get_transforms(data = "train"))
    valid_dataset = GetDataset(valid_folds,path = "./Dataset/train_images", transform=get_transforms(data = "valid"))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    epochs = 2
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=3, verbose=True)

    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        for images, labels in tqdm(train_loader):
          optimizer.zero_grad()
       
        # snapmix
          images_mix, target_a, target_b, lam_a, lam_b = snapmix(images, labels,5, resnext_cam)
          outputs = resnext(images_mix)
          loss_a = criterion(outputs, target_a)
          loss_b = criterion(outputs, target_b)
          loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
          loss.backward()
          optimizer.step()
    
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = resnext(images)
                loss = criterion(outputs, labels)
                loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        loss /= len(valid_loader)
        val_losses.append(loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            print(f"Epoch {epoch+1} - Save Best Score: {best_acc:.4f} Model")
            torch.save(
                {"model": resnext.state_dict()}, "./task/"+ f"{'resnext_snapmix'}_fold{fold}_best.pth"
            )
        if early_stopping(loss):
            print("Early stopping")
            break
    
    return val_losses, val_accuracies

def train_vit(train_copy,fold):
    #load model
    vit = timm.create_model('vit_base_patch16_384', pretrained=True)
    n_features = vit.head.in_features
    vit.head = nn.Linear(n_features, 5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit.parameters(), lr=0.0001)

    print(f"========== fold: {fold} training ==========")

    train_idx = train_copy[train_copy["fold"] != fold].index
    val_idx = train_copy[train_copy["fold"] == fold].index

    train_folds = train_copy.loc[train_idx].reset_index(drop=True)
    valid_folds = train_copy.loc[val_idx].reset_index(drop=True)

    train_dataset = GetDataset(train_folds,path = "./Dataset/train_images", transform=get_transforms(data = "train"))
    valid_dataset = GetDataset(valid_folds,path = "./Dataset/train_images", transform=get_transforms(data = "valid"))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    epochs = 2
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=3, verbose=True)

    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        for images, labels in tqdm(train_loader):
          optimizer.zero_grad()
          outputs = vit(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
    
    
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = vit(images)
                loss = criterion(outputs, labels)
                loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        loss /= len(valid_loader)
        val_losses.append(loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            print(f"Epoch {epoch+1} - Save Best Score: {best_acc:.4f} Model")
            torch.save(
                {"model": vit.state_dict()}, "./task/"+ f"{'vit'}_fold{fold}_best.pth"
            )
        if early_stopping(loss):
            print("Early stopping")
            break
    
    return val_losses, val_accuracies


def train_model(train):
    val_losses_list = []
    val_accuracies_list = []

    for fold in range(5):
       val_losses, val_accuracies = train_loop(train,fold)
       val_losses_list.append(val_losses)
       val_accuracies_list.append(val_accuracies)

    res_df = pd.DataFrame({
       'fold': [fold for fold in range(5)],
       'val_losses': val_losses_list,
       'val_accuracies': val_accuracies_list
    })

    res_df.to_csv('./task/validation_results2.csv', index=False)


def train_vit_model(train):
    val_losses_list = []
    val_accuracies_list = []

    for fold in range(5):
       val_losses, val_accuracies = train_vit(train,fold)
       val_losses_list.append(val_losses)
       val_accuracies_list.append(val_accuracies)

    res_df = pd.DataFrame({
       'fold': [fold for fold in range(5)],
       'val_losses': val_losses_list,
       'val_accuracies': val_accuracies_list
    })

    res_df.to_csv('./task/vit_results.csv', index=False)

def train_snapmix_model(train):
    val_losses_list = []
    val_accuracies_list = []

    for fold in range(5):
       val_losses, val_accuracies = train_snapmix(train,fold)
       val_losses_list.append(val_losses)
       val_accuracies_list.append(val_accuracies)

    res_df = pd.DataFrame({
       'fold': [fold for fold in range(5)],
       'val_losses': val_losses_list,
       'val_accuracies': val_accuracies_list
    })

    res_df.to_csv('./task/resnext_snapmix.csv', index=False)

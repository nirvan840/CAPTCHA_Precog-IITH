# general
from tqdm import tqdm
# time 
from datetime import datetime
import pytz

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights

# module
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str = 'simple', num_classes: int = 100) -> nn.Module:
    """
    Build and return a CNN model.

    Args:
        model_name: 'simple' for a 5-layer CNN or 'resnet50' for pretrained ResNet-50.
        num_classes: Number of output classes.
    """
    if model_name == 'simple': 
        # Simple 5-layer CNN ~10M parameters
        layers = []
        in_channels = 3
        channels = [64, 128, 128]               # NOTE 
        # Feature Extractor
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=config.get('dropout', 0)))
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        # FC Laeyr
        # layers.append(nn.Linear(channels[-1], channels[-1]//2))
        # layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=config.get('dropout', 0)))
        layers.append(nn.Linear(channels[-1], num_classes))
        return nn.Sequential(*layers)

    elif model_name == 'resnet50':
        # Load pretrained ResNet-50
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last two layers (layer3, layer4) + fc
        for name, param in model.named_parameters():
            if name.startswith('layer3') or name.startswith('layer4') or name.startswith('fc'):
                param.requires_grad = True
        # Replace final FC
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            total_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels)
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = corrects.double() / total_samples * 100
    return avg_loss, avg_acc


def train_and_evaluate(model: nn.Module,
                       config: dict) -> None:
    """
    Train and evaluate the given model using hyperparameters in config.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        test_easy_loader: DataLoader for easy test set.
        test_medium_loader: DataLoader for medium test set.
        config: Dictionary of hyperparameters and wandb config.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nTrainable parameters: {count_trainable_parameters(model):,}")

    # wandb
    if config.get('log_wandb', True): 
        import wandb
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb'].get('run_name'),
            config=config
        )

    # optimizer and scheduler
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config.get('min_lr', 0)
    )

    # loss
    criterion = nn.CrossEntropyLoss()
    best_test_loss = float('inf')
    save_path = config.get('save_path', 'best_model.pth')

    # dataloaders config
    print("")
    batch_size = config.get('batch_size', 32)
    is_resnet = (config['model'] == 'resnet50')
    if is_resnet:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std  = [0.5, 0.5, 0.5]
    # data transformation      
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # dataset
    train_data_root = config.get('train_data_root')
    test_data_root = config.get('test_data_root')
    tr_dataset = datasets.ImageFolder(root=train_data_root, transform=data_tf)
    tst_dataset = datasets.ImageFolder(root=test_data_root, transform=data_tf)
    # dataloader
    train_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader  = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    print(f"\nClass to Index mapping: {tr_dataset.class_to_idx}\n")
    
    # NOTE
    # model output shape: [batch_size, 100]
    # target shape: [batch_size] with values in [0, 99]
    # loss = nn.CrossEntropyLoss()
    # output = model(inputs)
    # loss_val = loss(output, target)
        
    # train
    print("")
    for epoch in tqdm(range(config['num_epochs']), total=config['num_epochs'], desc="Training", unit="epochs"):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            # DEBUG
            # print(outputs.shape)
            # print(labels.shape)
            # while True: pass
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        scheduler.step()

        if (epoch + 1) % config['log_epochs'] == 0:
            if config.get('log_wandb', True): 
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_acc.item(),
                    'lr': scheduler.get_last_lr()[0]
                })

        if (epoch + 1) % config['eval_epochs'] == 0:
            model.eval()
            # Evaluate on test set
            avg_test_loss, avg_test_acc = evaluate_model(model, test_loader, criterion, device)
         
            if config.get('log_wandb', True): 
                wandb.log({
                    'epoch': epoch + 1,
                    'test_easy_loss': avg_test_loss,
                    'test_easy_accuracy': avg_test_acc.item(),
                })

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), save_path)
                if config.get('log_wandb', True): 
                    wandb.log({'best_model_saved': True})
    
    if config.get('log_wandb', True): 
        wandb.finish()



if __name__ == "__main__":

    # Define IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    # Get current time in IST
    ist_time = datetime.now(ist)
    # Format as string
    ist_str = ist_time.strftime("%d-%m-%Y %H:%M:%S")
        
    # Special
    model_type = 'simple'
    mode = 'bonus'
    dropout = 0
    
    config = {
        # data 
        'train_data_root': 'task1_classification/data/train',
        'test_data_root': 'task1_classification/data/test',
        'batch_size': 512,  
        
        # cnn type (simple, resnet50)
        'model': model_type,
        'dropout': dropout,             # only for simple model
        
        # training    
        'lr':     1e-3,
        'min_lr': 1e-5,
        'num_epochs': 150,
        'log_epochs':  2,
        'eval_epochs': 2,
        'save_path': f'task2_generation/models/{mode}-{model_type}.pth',
        
        # logging
        'log_wandb': False, 
        'wandb': {
            'project': 'Precog-CNN',
            'api_key': '6273bda0322e6f5b38b888c6a9357d1cabd2ddf6',
            'run_name': f'{mode}-{model_type}_{ist_str}'
        }
    }
    
    # bonus (green vs red)
    if mode == 'bonus': num_cls = 2
    elif mode == 'characters': num_cls = 62
    
    # train
    model = build_model(config['model'], num_classes=num_cls)
    train_and_evaluate(model, config)
    print(f"\nTraining for {mode} completed")

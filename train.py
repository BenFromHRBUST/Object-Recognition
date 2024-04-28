import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix

import wandb

from tqdm import tqdm

import models
from dataset.CIFAR100 import CIFAR100
from utils import (check_device,
                   plot_curves,
                   draw_confusion_matrix)


def train_model(model, train_general_config, train_loader, val_loader, device='cpu', is_production=False):
    print(
        f'[+] Training {model.__class__.__name__} with {train_general_config["optimizer"]} optimizer for {train_general_config["epochs"]} epochs...')

    criterion = nn.CrossEntropyLoss()
    optimizer_class = getattr(optim, train_general_config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=train_general_config['learning_rate'], weight_decay=train_general_config['weight_decay'])

    print(f'[+] Model Architecture: {model}')
    print(f'[+] Optimizer: {optimizer}')
    print(f'[+] Criterion: {criterion}')

    model.train()
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(train_general_config['epochs']):
        running_loss = 0.0
        total_correct = 0
        total_images = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{train_general_config["epochs"]}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = total_correct / total_images * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if len(val_loader) != 0:
            model.eval()
            val_running_loss = 0.0
            val_total_correct = 0
            val_total_images = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total_images += labels.size(0)
                    val_total_correct += (predicted == labels).sum().item()

                val_loss = val_running_loss / len(val_loader)
                val_accuracy = val_total_correct / val_total_images * 100
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss,
                       "val_accuracy": val_accuracy}) if is_production else None
            print(
                f'Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.5f}%, Val Loss: {val_loss:.5f}, Val Accuracy: {val_accuracy:.5f}%')
        else:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy}) if is_production else None
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.5f}%')

    print(f'[+] Training {model.__class__.__name__} with {train_general_config["optimizer"]} optimizer for {train_general_config["epochs"]} epochs...DONE!')
    return train_losses, train_accuracies, val_losses, val_accuracies


def test_model(model, train_config, test_loader, device='cpu', is_production=False):
    print(f'[+] Testing {model.__class__.__name__}...')

    model.eval()
    total_correct = 0
    total_images = 0
    total_predictions = torch.tensor([]).to(device)
    total_labels = torch.tensor([]).to(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_predictions = torch.cat((total_predictions, predicted), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)
    cm = confusion_matrix(total_labels.cpu().numpy(), total_predictions.cpu().numpy())

    accuracy = total_correct / total_images * 100
    wandb.log({"Test Accuracy": accuracy}) if is_production else None
    print(f'Test Accuracy: {accuracy}%')

    print(f'[+] Testing {model.__class__.__name__}...DONE!')

    return cm


def train(model_name, train_general_config, train_model_config, dataset_name, dataset_config, is_production=False):
    dataset = eval(dataset_name)(dataset_config)
    train_loader, val_loader, test_loader = dataset.get_loader(train_general_config['batch_size'])

    device = check_device()

    model = getattr(models, model_name)(train_model_config).to(device)

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(model,
                                                                             train_general_config,
                                                                             train_loader,
                                                                             val_loader,
                                                                             device=device,
                                                                             is_production=is_production)

    plot_curves(train_losses, train_accuracies, val_losses, val_accuracies,
                title=f'{model_name}-{train_general_config["optimizer"]}',
                path=train_general_config['fig_path'])

    cm = test_model(model,
                    train_general_config,
                    test_loader,
                    device=device,
                    is_production=is_production)

    draw_confusion_matrix(cm,
                          title=f'ConfusionMatrix-{model_name}-{train_general_config["optimizer"]}',
                          path=train_general_config['fig_path'])

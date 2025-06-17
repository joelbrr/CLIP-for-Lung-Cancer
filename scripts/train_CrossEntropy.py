# -*- coding: utf-8 -*-
"""
train_CrossEntropy.py – v3 (20 May 2025)

• **Tres** rutinas de entrenamiento independientes:
  1. `train_retrieval`   – contraste imagen ↔ texto (InfoNCE)
  2. `train_classifier`  – clasificación de descriptor (CrossEntropy)
  3. `train_regressor`   – regresión de descriptor continuo (MSE / MAE)

Todas comparten helpers de visualización (gráficas con texto grande).
"""

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Ajustes globales de gráficos (fuentes grandes)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})
sns.set(font_scale=1.2)

# -----------------------------------------------------------------------------
# UTILIDADES VISUALES
# -----------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _plot_losses(losses: list[float], title: str):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 1) CONTRASTIVE (RETRIEVAL)
# -----------------------------------------------------------------------------

def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    return nn.CrossEntropyLoss()(logits, labels)


def train_retrieval(model, train_loader, optimizer, device, epochs: int = 5):
    model.train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        all_preds, all_labels = [], []

        print(f"— Retrieval | Epoch {epoch}/{epochs} —")
        for img_feats, reports, _ in tqdm(train_loader, leave=False):
            img_feats = img_feats.to(device)
            reports = {k: v.squeeze(1).to(device) for k, v in reports.items()}

            logits = model(img_feats, reports)
            loss = _contrastive_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            labels = list(range(len(preds)))
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"  Loss: {avg_loss:.6f}\n")
        _plot_confusion_matrix(all_labels, all_preds)

    _plot_losses(epoch_losses, "Retrieval Loss per epoch")
    return epoch_losses

# -----------------------------------------------------------------------------
# 2) CLASSIFICATION
# -----------------------------------------------------------------------------

def train_classifier(
    model,
    train_loader,
    optimizer,
    device,
    epochs: int = 10,
    criterion: nn.Module | None = None,
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        all_preds, all_labels = [], []

        print(f"— Classifier | Epoch {epoch}/{epochs} —")
        for images, labels in tqdm(train_loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(f"  Loss: {avg_loss:.4f} | Acc: {acc:.3f} | F1-w: {f1:.3f}\n")
        _plot_confusion_matrix(all_labels, all_preds, title="Train Confusion Matrix")

    _plot_losses(epoch_losses, "Classifier Loss per epoch")
    return epoch_losses

# -----------------------------------------------------------------------------
# 3) REGRESSION
# -----------------------------------------------------------------------------

def _rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def train_regressor(
    model,
    train_loader,
    optimizer,
    device,
    epochs: int = 10,
    criterion: nn.Module | None = None,
):
    """Entrena un modelo para predecir un valor continuo (MSE)."""
    if criterion is None:
        criterion = nn.MSELoss()

    model.train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        total_loss, y_true_all, y_pred_all = 0.0, [], []

        print(f"— Regressor | Epoch {epoch}/{epochs} —")
        for images, targets in tqdm(train_loader, leave=False):
            images = images.to(device)
            targets = targets.float().to(device)

            preds = model(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            y_true_all.extend(targets.cpu().tolist())
            y_pred_all.extend(preds.detach().cpu().tolist())

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        rmse = _rmse(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)

        print(f"  MSE: {avg_loss:.4f} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}\n")

    _plot_losses(epoch_losses, "Regressor MSE per epoch")
    return epoch_losses

    
    
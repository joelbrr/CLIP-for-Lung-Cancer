# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:08:09 2025

@author: joelb
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
from evaluation.evaluate import qualitative_evaluation


def contrastive_loss(logits):
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size).to(logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def create_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def train_model(model, train_loader, optimizer, device, epochs=5, tokenizer=None):
    model.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        print(f"---- Epoch {epoch+1} ----")

        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader):
            image_features, reports, patient_ids = batch
            image_features = image_features.to(device)
            reports = {k: v.squeeze(1).to(device) for k, v in reports.items()}

            logits = model(image_features, reports)

            loss = contrastive_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                labels = torch.arange(len(preds)).to(device)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}] Loss: {avg_loss:.6f}")

        create_confusion_matrix(all_labels, all_preds)

    plt.plot(range(1, epochs + 1), losses, label="Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
    
    
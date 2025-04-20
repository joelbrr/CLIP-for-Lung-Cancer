# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:42:37 2025

@author: joelb
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


# --- Modelo CLIP-based (Imagen + Texto) ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim=768):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

    def forward(self, word_embeddings):
        attn_output, attn_weights = self.attention(word_embeddings, word_embeddings, word_embeddings)
        return attn_output, attn_weights

class TextEncoder(nn.Module):
    def __init__(self, model_name="medicalai/ClinicalBERT", feature_dim=512):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Congelar las capas del transformer
        #for param in self.transformer.parameters():
         #   param.requires_grad = False
        
        # Proyección entrenable de cada token embedding (768 → 512)
        self.word_projection = nn.Linear(self.transformer.config.hidden_size, feature_dim)

        # Si quieres mantener atención: cámbiala a 512 también
        self.attention_layer = AttentionLayer(hidden_dim=feature_dim)

    def forward(self, input_text):
        input_ids = input_text['input_ids'].squeeze(1)           # [B, seq_len]
        attention_mask = input_text['attention_mask'].squeeze(1) # [B, seq_len]

        # Salida de BERT (ya está congelado, no se actualizan sus parámetros)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state              # [B, seq_len, 768]

        # Proyección token a token → [B, seq_len, 512]
        projected = self.word_projection(word_embeddings)

        # Atención (opcional, pero ahora trabaja en 512d)
        attn_output, _ = self.attention_layer(projected)         # [B, seq_len, 512]

        # 👉 Retornar todos los embeddings de tokens
        return attn_output  # [B, seq_len, 512]




# --- Modelo multimodal CLIPMedical ---
class CLIPMedical(nn.Module):
    def __init__(self, feature_dim=512):
        super(CLIPMedical, self).__init__()
        self.text_encoder = TextEncoder(feature_dim=feature_dim)
        self.image_projection = nn.Sequential(
            nn.Linear(1024, feature_dim), 
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        #for param in self.image_projection.parameters():
        #       param.requires_grad = False
                
    def forward(self, image_features, texts):
        text_embeddings = self.text_encoder(texts)  # [B, seq_len, 512]
        image_embeddings = self.image_projection(image_features)  # [B, 512]

        # Normalizar
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=2, keepdim=True)

        # Expandir imagen para comparar con cada token
        image_embeddings = image_embeddings.unsqueeze(1)  # [B, 1, 512]

        # Producto punto imagen-token → similitud por token
        #similarities = (image_embeddings * text_embeddings).sum(dim=2)  # [B, seq_len]

        # Estrategia: usar el máximo de similitud por texto
        #logits = similarities.max(dim=1).values  # [B] → logit por texto

        # Expandimos para calcular cross entropy pairwise (imagen ↔ texto)
        logits_matrix = torch.cdist(
            image_embeddings.squeeze(1),  # [B, 512]
            text_embeddings.mean(dim=1),  # [B, 512] baseline
            p=2 
        )
        logits_matrix = -logits_matrix * torch.exp(self.temperature)  # menor distancia = más parecido
        """for i in range(logits_matrix.shape[0]):
            for j in range(logits_matrix.shape[1]):
                print(f"Sim(image {i} ↔ text {j}): {logits_matrix[i, j].item():.3f}") 
                """

        return logits_matrix

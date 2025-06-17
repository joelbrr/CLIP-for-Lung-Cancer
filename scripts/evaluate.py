# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:14:39 2025

@author: joelb
"""

# evaluate.py
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from collections import defaultdict

clinical_fields = ["nodule shape", "nodule density", "infiltration", "cdiff", "necrosis"]

def extract_fields(text):
    text = text.lower()
    fields = {}
    for field in clinical_fields:
        match = re.search(rf"{field}\s*:\s*([^,]+)", text)
        fields[field] = match.group(1).strip() if match else "unknown"
    return fields

def qualitative_evaluation(image_embeddings, text_embeddings, patient_ids, text_input_ids, tokenizer, top_k=5):
    text_embeddings = text_embeddings.mean(axis=1)
    similarities = cosine_similarity(image_embeddings, text_embeddings)

    for i, pid in enumerate(patient_ids):
        top_k_indices = similarities[i].argsort()[-top_k:][::-1]
        real_text_index = np.where(patient_ids == pid)[0][0]
        print(f"\nPatient {pid} - Top {top_k} most similar texts:")
        if real_text_index in top_k_indices:
            print(f"✅ Correct! Real text is in the top {top_k}.")
        else:
            print(f"❌ Incorrect! Real text is NOT in the top {top_k}.")

        for idx in top_k_indices:
            decoded_text = tokenizer.decode(text_input_ids[idx], skip_special_tokens=True)
            print(f"Text {idx}: {decoded_text} with similarity: {similarities[i][idx]:.3f}")

        decoded_real_text = tokenizer.decode(text_input_ids[real_text_index], skip_special_tokens=True)
        print(f"Real Text for Patient {pid}: {decoded_real_text}")

def evaluate_fields(test_ids, test_real_texts, top_k_predictions):
    total_by_field = defaultdict(int)
    correct_by_field = defaultdict(int)

    for i, pid in enumerate(test_ids):
        real_text = test_real_texts[i]
        real_fields = extract_fields(real_text)
        top_k_texts = top_k_predictions[i]

        for pred_text in top_k_texts:
            pred_fields = extract_fields(pred_text)
            for field in clinical_fields:
                total_by_field[field] += 1
                if real_fields[field] == pred_fields[field]:
                    correct_by_field[field] += 1

    print("\n📊 Accuracy por campo:")
    for field in clinical_fields:
        total = total_by_field[field]
        correct = correct_by_field[field]
        acc = correct / total if total else 0
        print(f"  - {field.title()}: {acc:.2%}")
        
def visualize_attention_map(input_ids, attn_weights, tokenizer, title="Attention Weights", focus_token_idx=0):
    """
    Visualiza el mapa de atención promedio sobre los tokens de entrada.
    - input_ids: [seq_len] tensor
    - attn_weights: [batch_size, num_heads, seq_len, seq_len] o [1, seq_len, seq_len]
    - focus_token_idx: índice del token en el que se quiere centrarse (por ejemplo, 0 para [CLS])
    """
    # Imprimir las dimensiones de attn_weights para depuración
    print(f"Dimensiones de attn_weights: {attn_weights.shape}")
    
    # Comprobar si las dimensiones de attn_weights son correctas
    if attn_weights.dim() == 3:
        print("⚠️ Atención tiene 3 dimensiones (un solo cabezal de atención), ajustando para visualización.")
        attn_weights = attn_weights.unsqueeze(0)  # Convertir a 4 dimensiones [1, seq_len, seq_len]
    
    # Verificación si aún no tenemos las 4 dimensiones necesarias
    if attn_weights.dim() != 4:
        print(f"⚠️ Atención no tiene 4 dimensiones. Dimensiones actuales: {attn_weights.shape}. No se puede visualizar.")
        return

    # Si solo hay un batch, seleccionamos el primer batch (atención para todo el batch)
    if attn_weights.shape[0] == 1:
        attn_weights = attn_weights[0]  # [num_heads, seq_len, seq_len] (tomamos el primer batch)
    else:
        print(f"⚠️ Atención con más de un batch no está soportada en esta visualización.")
        return

    avg_attn = attn_weights.mean(dim=0)  # Promedio sobre las cabezas de atención [seq_len, seq_len]

    # Si queremos observar la atención de un token específico (por ejemplo, el primer token [CLS])
    token_attn = avg_attn[focus_token_idx]  # Atención a cada token respecto al token de foco

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Visualización en un gráfico
    plt.figure(figsize=(12, 1.5))
    sns.heatmap(token_attn.unsqueeze(0).cpu().numpy(), annot=np.array(tokens).reshape(1, -1),
                fmt='', cmap='Blues', cbar=False)
    plt.title(f"{title} - Focusing on Token {tokens[focus_token_idx]}")
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def confusion_matrix_by_field(test_ids, real_texts_dict, top_k_texts_predicted):
    clinical_fields = ["nodule shape", "nodule density", "infiltration", "cdiff","necrosis"]

    for field in clinical_fields:
        y_true = []
        y_pred = []

        for i, pid in enumerate(test_ids):
            real_text = real_texts_dict[pid]
            pred_text = top_k_texts_predicted[i][0]  # solo top-1

            real_fields = extract_fields(real_text)
            pred_fields = extract_fields(pred_text)

            y_true.append(real_fields[field])
            y_pred.append(pred_fields[field])

        labels = sorted(set(y_true + y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        print(f"\n📊 Confusion Matrix for field: {field.upper()}")
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        for label in ax.texts:  # Iterar sobre las anotaciones de texto (números)
            label.set_fontsize(20)  # Establecer el tamaño de la fuente para los números
        plt.title(f"Confusion Matrix - {field.upper()}")
        plt.show()
        
        
def evaluate_fold_metrics(test_ids, real_texts_dict, top_k_texts_predicted, fold_index=0):
    y_true_by_field = defaultdict(list)
    y_pred_by_field = defaultdict(list)

    for i, pid in enumerate(test_ids):
        real_text = real_texts_dict[pid]
        pred_text = top_k_texts_predicted[i][0]  # solo top-1

        real_fields = extract_fields(real_text)
        pred_fields = extract_fields(pred_text)

        for field in clinical_fields:
            y_true_by_field[field].append(real_fields[field])
            y_pred_by_field[field].append(pred_fields[field])

    print(f"\n📊 Métricas para Fold {fold_index + 1}")
    metrics_by_field = {}

    for field in clinical_fields:
        y_true = y_true_by_field[field]
        y_pred = y_pred_by_field[field]
        labels = sorted(list(set(y_true + y_pred)))

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"\n🔹 Variable: {field.title()}")
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1 Score:  {f1:.3f}")

        metrics_by_field[field] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }

    return metrics_by_field


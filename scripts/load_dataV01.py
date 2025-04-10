# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 22:17:19 2025

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




# --- Configuración de rutas y parámetros ---
DataDir = r'C:\Users\joelb\OneDrive\Escritorio\4t\TFG\data\CanRuti'
npz_file_path_glcm = os.path.join(DataDir, 'ResNet18_slice_GLCM3D.npz')
npz_file_path_intensity = os.path.join(DataDir, 'ResNet18_slice_intensity.npz')
excel_meta_path = os.path.join(DataDir, '11112024_BDMetaData_CanRuti_RadDescriptors (1).xlsx')

# --- Carga de datos clínicos ---
# --- Carga de datos clínicos ---
def LoadCanRutiClinicalData(excel_path, tokenizer):
    df = pd.read_excel(excel_path)

    # Usamos directamente patient_id como CaseID (int) para facilitar el mapeo
    df['CaseID'] = df['patient_id'].astype(int)
    df = df.sort_values(by='CaseID').reset_index(drop=True)

    patient_id_to_text = {}

    for _, row in df.iterrows():
        pid = int(row['CaseID'])

        # Generar texto clínico
        text = (
            f"Nodule Shape: {row.get('nodule_shape', 'unknown')}, "
            f"Nodule Density: {row.get('nodule_density', 'unknown')}, "
            f"Infiltration: {row.get('vinfiltration', 'unknown')}, "
            f"Cdiff: {row.get('cdiff', 'unknown')}, "
            f"Necrosis: {row.get('necrosis', 'unknown')}"
        )

        # Tokenizar y guardar por ID
        patient_id_to_text[pid] = tokenizer(
            text, padding='max_length', truncation=True, max_length=256, return_tensors="pt"
        )

    return patient_id_to_text

# --- Tokenizador ---
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

# --- Cargar textos clínicos tokenizados por paciente ---
patient_id_to_text = LoadCanRutiClinicalData(
    excel_path=excel_meta_path,
    tokenizer=tokenizer
)

# --- Carga de características de imagen y agrupación por paciente ---
def extract_patient_id(patient_id_str):
    if isinstance(patient_id_str, np.ndarray):
        patient_id_str = patient_id_str[0]
    return patient_id_str[5:].split('_')[0]  # Devuelve string tipo '70'

data_glcm = np.load(npz_file_path_glcm, allow_pickle=True)
data_intensity = np.load(npz_file_path_intensity, allow_pickle=True)
slice_features_glcm = data_glcm['slice_features']
slice_features_intensity = data_intensity['slice_features']
slice_meta = data_glcm['slice_meta']

patient_slices = {}
current_patient_id = None
current_indices = []

for idx in range(slice_meta.shape[0]):
    pid = int(extract_patient_id(slice_meta[idx][0]))  # Convertir a int
    if pid != current_patient_id:
        if current_patient_id is not None:
            patient_slices[current_patient_id] = current_indices
        current_patient_id = pid
        current_indices = [idx]
    else:
        current_indices.append(idx)
if current_patient_id is not None:
    patient_slices[current_patient_id] = current_indices

# --- Pooling y fusión de características de imagen ---
pooled_image_features = []
patient_keys_ordered = []

for patient_id, indices in patient_slices.items():
    glcm = slice_features_glcm[indices].mean(axis=1).mean(axis=0)  # (512,)
    intensity = slice_features_intensity[indices].squeeze(axis=1).mean(axis=0)  # (512,)
    fused = np.concatenate([glcm, intensity], axis=0)  # (1024,)
    pooled_image_features.append(fused)
    patient_keys_ordered.append(patient_id)

# --- Mapeo explícito paciente → imagen ---
patient_id_to_image = dict(zip(patient_keys_ordered, pooled_image_features))

# --- Filtrar pacientes comunes en imagen y texto ---
common_ids = [pid for pid in patient_keys_ordered if pid in patient_id_to_text]
print("Pacientes comunes:", len(common_ids))


all_patient_ids = list(set(patient_keys_ordered) & set(common_ids))
train_ids, test_ids = train_test_split(all_patient_ids, test_size=0.2, random_state=42)
# --- Construcción de listas alineadas ---
# TRAIN
train_image_features = [patient_id_to_image[pid] for pid in train_ids]
train_text_features  = [patient_id_to_text[pid] for pid in train_ids]
train_patient_ids    = train_ids

# TEST
test_image_features = [patient_id_to_image[pid] for pid in test_ids]
test_text_features  = [patient_id_to_text[pid] for pid in test_ids]
test_patient_ids    = test_ids


# --- Dataset y DataLoader ---
class MultimodalDataset(Dataset):
    def __init__(self, image_features, tokenized_texts, patient_ids):
        self.image_features = torch.tensor(image_features, dtype=torch.float32)
        self.tokenized_texts = tokenized_texts
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        image_feature = self.image_features[idx]
        text_description = self.tokenized_texts[idx]
        patient_id = self.patient_ids[idx]
        return image_feature, text_description, patient_id

train_dataset = MultimodalDataset(train_image_features, train_text_features, train_patient_ids)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)



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
        for param in self.transformer.parameters():
            param.requires_grad = False
        
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
        
        for param in self.image_projection.parameters():
                param.requires_grad = False

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


def qualitative_evaluation(image_embeddings, text_embeddings, patient_ids, text_input_ids, top_k=5):
    # Promediar los embeddings de texto a lo largo de la secuencia (seq_len) para que tenga la misma forma que el de imagen
    text_embeddings = text_embeddings.mean(axis=1)  # Promediar a lo largo de seq_len: [batch_size, feature_dim]
    
    # Asegurarnos de que ambos conjuntos de embeddings tienen la misma forma
    print(f"Longitudes después del promedio de texto: Imagen: {image_embeddings.shape}, Texto: {text_embeddings.shape}")

    # Calcula la similitud de coseno entre cada imagen y todos los textos
    similarities = cosine_similarity(image_embeddings, text_embeddings)

    # Imprimir las mejores predicciones (más similares)
    for i, patient_id in enumerate(patient_ids):
        # Top-K más similares para la imagen de este paciente
        top_k_indices = similarities[i].argsort()[-top_k:][::-1]  # Indices de los K más similares
        print(f"\nPatient {patient_id} - Top {top_k} most similar texts:")

        # Encontrar el índice del texto real en patient_ids
        real_text_index = np.where(patient_ids == patient_id)[0][0]  # Buscar el índice usando np.where

        # Mostrar si el texto real está entre los top k más similares
        if real_text_index in top_k_indices:
            print(f"✅ Correct! Real text is in the top {top_k}.")
        else:
            print(f"❌ Incorrect! Real text is NOT in the top {top_k}.")

        # Ahora mostramos los textos más similares junto con su similitud
        for idx in top_k_indices:
            # Decodificar los textos más similares usando los input_ids del tokenizador
            decoded_text = tokenizer.decode(text_input_ids[idx], skip_special_tokens=True)
            print(f"Text {idx}: {decoded_text} with similarity: {similarities[i][idx]:.3f}")

        # Mostrar el texto real para el paciente en cuestión
        decoded_real_text = tokenizer.decode(text_input_ids[real_text_index], skip_special_tokens=True)
        print(f"Real Text for Patient {patient_id}: {decoded_real_text}")





# --- Pérdida Contrastiva ---
def contrastive_loss(logits):
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size).to(logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels) #+ nn.CrossEntropyLoss()(logits.T, labels)
    return loss

# --- Entrenamiento ---
def train_model(model, train_loader, optimizer, device, epochs=5):
    model.train()
    losses = []  # Lista para almacenar las pérdidas

    for epoch in range(epochs):
        total_loss = 0
        print(f"---- Epoch {epoch+1} ----")
        image_embeddings = []
        text_embeddings = []
        patient_ids_list = []  # Lista para almacenar los patient_ids del batch
        text_input_ids_list = []  # Lista para almacenar los input_ids de los textos

        for batch in train_loader:
            image_features, reports, patient_ids = batch

            image_features = image_features.to(device)
            reports = {k: v.squeeze(1).to(device) for k, v in reports.items()}

            # Extraer embeddings de imagen y texto
            image_embeddings_batch = model.image_projection(image_features).cpu().detach().numpy()
            text_embeddings_batch = model.text_encoder(reports).cpu().detach().numpy()

            # Guardar los embeddings y patient_ids del batch
            image_embeddings.append(image_embeddings_batch)
            text_embeddings.append(text_embeddings_batch)
            patient_ids_list.append(patient_ids.cpu().detach().numpy())

            # Obtener los input_ids de los textos (necesarios para la evaluación cualitativa)
            text_input_ids_list.append(reports['input_ids'].cpu().detach().numpy())

            # Calcular logits
            logits = model(image_features, reports)
            """
            plt.figure(figsize=(8, 6))
            sns.heatmap(logits.detach().cpu().numpy(), cmap="viridis", annot=True, fmt=".2f")
            plt.xlabel("Textos (posición en batch)")
            plt.ylabel("Imágenes (posición en batch)")
            plt.title("🔍 Similitud imagen ↔ texto (logits)")
            plt.show()"""

            # Calcular pérdida
            loss = contrastive_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Métrica de precisión por batch
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                labels = torch.arange(len(preds)).to(device)
                acc = (preds == labels).float().mean()
                print(f"Batch accuracy: {acc.item():.2f}")

        # Convertir listas de embeddings y patient_ids a arrays de NumPy
        image_embeddings = np.vstack(image_embeddings)
        text_embeddings = np.vstack(text_embeddings)
        patient_ids_list = np.concatenate(patient_ids_list)  # Unir todos los patient_ids del batch
        text_input_ids_list = np.concatenate(text_input_ids_list)  # Unir todos los input_ids

        # --- Evaluación cualitativa ---
        qualitative_evaluation(image_embeddings, text_embeddings, patient_ids_list, text_input_ids_list)

        # Promediar la pérdida
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}] Loss: {avg_loss:.6f}")

    # Graficar la evolución de la pérdida
    plt.plot(range(1, epochs + 1), losses, label="Pérdida por época")
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la Pérdida durante el Entrenamiento')
    plt.legend()
    plt.show()



# --- Inicializar el modelo y el optimizador ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPMedical().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Entrenamiento ---
train_model(model, train_loader, optimizer, device, epochs=40)

# Construir catálogo de texto (solo TEST)
train_texts = [patient_id_to_text[pid] for pid in train_ids]
train_catalog_ids = train_ids


with torch.no_grad():
    raw_train_texts = [
        f"Nodule Shape: {row.get('nodule_shape', 'unknown')}, "
        f"Nodule Density: {row.get('nodule_density', 'unknown')}, "
        f"Infiltration: {row.get('vinfiltration', 'unknown')}, "
        f"Cdiff: {row.get('cdiff', 'unknown')}, "
        f"Necrosis: {row.get('necrosis', 'unknown')}"
        for pid in train_ids
        for _, row in pd.read_excel(excel_meta_path).query("patient_id == @pid").iterrows()
    ]

    tokenized_catalog = tokenizer(
        raw_train_texts,
        padding='max_length', truncation=True, max_length=256, return_tensors="pt"
    )

    tokenized_catalog = {k: v.to(device) for k, v in tokenized_catalog.items()}
    catalog_embeddings = model.text_encoder(tokenized_catalog).mean(dim=1).cpu()  # [N, 512]

    torch.save({
        "embeddings": catalog_embeddings,
        "patient_ids": train_ids,
        "raw_texts": raw_train_texts
    }, "text_catalog_train.pt")
    
    
real_texts_dict = {
    pid: (
        f"Nodule Shape: {row.get('nodule_shape', 'unknown')}, "
        f"Nodule Density: {row.get('nodule_density', 'unknown')}, "
        f"Infiltration: {row.get('vinfiltration', 'unknown')}, "
        f"Cdiff: {row.get('cdiff', 'unknown')}, "
        f"Necrosis: {row.get('necrosis', 'unknown')}"
    )
    for pid in test_patient_ids
    for _, row in pd.read_excel(excel_meta_path).query("patient_id == @pid").iterrows()
}


clinical_fields = ["nodule shape", "nodule density", "infiltration", "cdiff", "necrosis"]

def extract_fields(text):
    """
    Extrae los campos clínicos clave de un texto clínico.
    Devuelve un diccionario {campo: valor}.
    """
    text = text.lower()
    fields = {}
    for field in clinical_fields:
        match = re.search(rf"{field}\s*:\s*([^,]+)", text)
        if match:
            fields[field] = match.group(1).strip()
        else:
            fields[field] = "missing"
    return fields

def evaluate_fields(test_ids, test_real_texts, catalog_texts, top_k_predictions, output_csv="evaluation_metrics.csv"):
    """
    Evalúa campo por campo si el texto recuperado top-K acierta los campos clínicos del texto real.
    """
    total_by_field = defaultdict(int)
    correct_by_field = defaultdict(int)
    
    # Para almacenar las métricas por paciente
    metrics = []

    for i, pid in enumerate(test_ids):
        real_text = test_real_texts[i]
        real_fields = extract_fields(real_text)

        top_k_texts = top_k_predictions[i]

        print(f"\n📌 Evaluando paciente {pid}")
        print(f"Real: {real_text}")

        for j, pred_text in enumerate(top_k_texts):
            pred_fields = extract_fields(pred_text)
            print(f"  🔍 Top {j+1} predicción: {pred_text}")

            for field in clinical_fields:
                total_by_field[field] += 1
                if real_fields[field] == pred_fields[field]:
                    correct_by_field[field] += 1
                    print(f"    ✔️ {field.title()} OK ({real_fields[field]})")
                else:
                    print(f"    ❌ {field.title()} Error (real: {real_fields[field]} | predicho: {pred_fields[field]})")
        
        # Almacenar las métricas para este paciente
        patient_metrics = {
            'Patient ID': pid,
            **{f'{field} Accuracy': correct_by_field[field] / total_by_field[field] if total_by_field[field] > 0 else 0 for field in clinical_fields}
        }
        metrics.append(patient_metrics)

    # Resultados globales por campo
    print("\n📊 Accuracy por campo:")
    for field in clinical_fields:
        total = total_by_field[field]
        correct = correct_by_field[field]
        acc = correct / total if total else 0
        print(f"  - {field.title()}: {acc:.2%}")

    # Guardar las métricas por paciente y por campo en un archivo CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_csv, index=False)
    print(f"\n📥 Métricas guardadas en '{output_csv}'")


top_k = 5
top_k_texts_predicted = []

for img_feat in test_image_features:
    img_tensor = torch.tensor(img_feat).unsqueeze(0).to(device)
    img_emb = model.image_projection(img_tensor).cpu().detach().numpy()  # [1, 512]

    sims = cosine_similarity(img_emb, catalog_embeddings.numpy())  # [1, N_train]
    top_idxs = sims[0].argsort()[::-1][:top_k]

    # Guardamos los textos predichos
    top_texts = [raw_train_texts[i] for i in top_idxs]
    top_k_texts_predicted.append(top_texts)



# Llamada a la función evaluate_fields con la nueva estructura para guardar las métricas en CSV
evaluate_fields(
    test_ids=test_patient_ids,                         # Lista de pacientes de test
    test_real_texts=[real_texts_dict[pid] for pid in test_patient_ids],  # Textos clínicos reales de test
    catalog_texts=raw_train_texts,                      # Textos del catálogo (usados en inferencia)
    top_k_predictions=top_k_texts_predicted,            # Lista de listas: textos recuperados top-k por imagen
    output_csv="evaluation_metrics_text&img_freeze.csv"                 # Especificar el nombre del archivo CSV donde se guardarán las métricas
)






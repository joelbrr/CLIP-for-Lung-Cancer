# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:45:55 2025

@author: joelb
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.model_selection import KFold
import glob


# --- Configuración de rutas y parámetros ---
#DataDir = r'C:\Users\joelb\OneDrive\Escritorio\4t\TFG\data\CanRuti'
#npz_file_path_glcm = os.path.join(DataDir, 'ResNet18_slice_GLCM3D.npz')
#npz_file_path_intensity = os.path.join(DataDir, 'ResNet18_slice_intensity.npz')
#excel_meta_path = os.path.join(DataDir, '11112024_BDMetaData_CanRuti_RadDescriptors (1).xlsx')

# --- Carga de datos clínicos ---
def LoadCanRutiClinicalData(excel_path, tokenizer):
    df = pd.read_excel(excel_path)

    # Usamos directamente patient_id como CaseID (int) para facilitar el mapeo
    df['CaseID'] = df['patient_id'].astype(int)
    df = df.sort_values(by='CaseID').reset_index(drop=True)

    patient_id_to_text = {}

    for _, row in df.iterrows():
        pid = int(row['CaseID'])

        necrosis_value = row.get('necrosis', 'unknown')
        if necrosis_value == 0.0:  # Si es 0.0, convertirlo a 0
            necrosis_value = 0
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



def extract_patient_id(patient_id_str):
    if isinstance(patient_id_str, np.ndarray):
        patient_id_str = patient_id_str[0]
    return int(patient_id_str[5:].split('_')[0])


# --- Tokenizador ---
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
"""
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
"""

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
"""
train_dataset = MultimodalDataset(train_image_features, train_text_features, train_patient_ids)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
"""


def get_fold_data(patient_id_to_image, patient_id_to_text, k_folds=5, current_fold=0, batch_size=16):
    """
    Devuelve train_loader y test_loader para el fold especificado.
    """

    assert 0 <= current_fold < k_folds, "Fold actual fuera de rango."

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_patient_ids = np.array(list(set(patient_id_to_image) & set(patient_id_to_text)))

    train_ids, test_ids = list(kf.split(all_patient_ids))[current_fold]
    train_ids = all_patient_ids[train_ids]
    test_ids = all_patient_ids[test_ids]

    # Construir los conjuntos
    train_image_features = [patient_id_to_image[pid] for pid in train_ids]
    train_text_features  = [patient_id_to_text[pid] for pid in train_ids]
    train_patient_ids    = train_ids

    test_image_features = [patient_id_to_image[pid] for pid in test_ids]
    test_text_features  = [patient_id_to_text[pid] for pid in test_ids]
    test_patient_ids    = test_ids

    # Dataset y DataLoader
    train_dataset = MultimodalDataset(train_image_features, train_text_features, train_patient_ids)
    test_dataset  = MultimodalDataset(test_image_features, test_text_features, test_patient_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ids.tolist(), test_ids.tolist()


class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: [S, D]
        scores = self.attn(x).squeeze(-1)         # [S]
        weights = F.softmax(scores, dim=0)        # [S]
        weighted = torch.sum(x * weights.unsqueeze(-1), dim=0)  # [D]
        return weighted

def load_multi_hospital_data(hospital_dirs, tokenizer, use_attention=True, device='cpu'):
    patient_id_to_image_all = {}
    patient_id_to_text_all = {}
    num_slices_target = 15
    expected_feature_dim = None

    for hospital_dir in hospital_dirs:
        excel_path = glob.glob(os.path.join(hospital_dir, "*.xlsx"))[0]
        npz_glcm = os.path.join(hospital_dir, 'MobileNetV2_slice_GLCM3D.npz')
        npz_intensity = os.path.join(hospital_dir, 'MobileNetV2_slice_intensity.npz')

        df = pd.read_excel(excel_path)
        df['CaseID'] = df['patient_id'].astype(int)
        df = df.sort_values(by='CaseID').reset_index(drop=True)

        for _, row in df.iterrows():
            pid = int(row['CaseID'])
            text = (
                f"Nodule Shape: {row.get('nodule_shape', 'unknown')}, "
                f"Nodule Density: {row.get('nodule_density', 'unknown')}, "
                f"Infiltration: {row.get('vinfiltration', 'unknown')}, "
                f"Cdiff: {row.get('cdiff', 'unknown')}, "
                f"Necrosis: {row.get('necrosis', 'unknown')}"
            )
            patient_id_to_text_all[pid] = tokenizer(
                text, padding='max_length', truncation=True, max_length=256, return_tensors="pt"
            )

        data_glcm = np.load(npz_glcm, allow_pickle=True)
        data_intensity = np.load(npz_intensity, allow_pickle=True)
        slice_features_glcm = data_glcm['slice_features']
        slice_features_intensity = data_intensity['slice_features']
        slice_meta = data_glcm['slice_meta']

        patient_slices = {}
        current_patient_id = None
        current_indices = []

        for idx in range(slice_meta.shape[0]):
            pid = extract_patient_id(slice_meta[idx][0])
            if pid != current_patient_id:
                if current_patient_id is not None:
                    patient_slices[current_patient_id] = current_indices
                current_patient_id = pid
                current_indices = [idx]
            else:
                current_indices.append(idx)
        if current_patient_id is not None:
            patient_slices[current_patient_id] = current_indices

        base_feature_dim = slice_features_glcm.shape[-1] + slice_features_intensity.shape[-1]
        aggregator = AttentionAggregator(base_feature_dim).to(device) if use_attention else None

        for pid, indices in patient_slices.items():
            if len(indices) < 1:
                continue

            mid = len(indices) // 2
            half = num_slices_target // 2
            start = max(mid - half, 0)
            end = start + num_slices_target
            if end > len(indices):
                end = len(indices)
                start = max(end - num_slices_target, 0)
            selected_indices = indices[start:end]

            glcm_feats = slice_features_glcm[selected_indices]
            if glcm_feats.ndim == 3:
                glcm_feats = glcm_feats.mean(axis=1)  # [15, 1280]

            print(f"PID {pid} - glcm_feats shape before squeeze: {glcm_feats.shape}")
            if glcm_feats.ndim == 3 and glcm_feats.shape[1] == 1:
                glcm_feats = glcm_feats.squeeze(axis=1)

            intensity_feats = slice_features_intensity[selected_indices]
            print(f"PID {pid} - intensity_feats shape before squeeze: {intensity_feats.shape}")
            if intensity_feats.ndim == 3 and intensity_feats.shape[1] == 1:
                intensity_feats = intensity_feats.squeeze(axis=1)

            print(f"PID {pid} - glcm_feats shape: {glcm_feats.shape}, intensity_feats shape: {intensity_feats.shape}")
            slice_feats = np.concatenate([glcm_feats, intensity_feats], axis=1)

            if slice_feats.shape[0] != num_slices_target:
                print(f"⚠️ Skipping patient {pid}: only {slice_feats.shape[0]} slices available")
                continue

            if use_attention:
                slice_tensor = torch.tensor(slice_feats, dtype=torch.float32).to(device)
                fused = aggregator(slice_tensor).detach().cpu().numpy()
            else:
                fused = slice_feats.mean(axis=0)

            # Comprobación de dimensión consistente
            if expected_feature_dim is None:
                expected_feature_dim = fused.shape[0]
            if fused.shape[0] != expected_feature_dim:
                print(f"❌ Skipping patient {pid}: feature length {fused.shape[0]} != expected {expected_feature_dim}")
                continue

            patient_id_to_image_all[pid] = fused

    print(f"✅ Total pacientes combinados: {len(set(patient_id_to_image_all) & set(patient_id_to_text_all))}")
    return patient_id_to_image_all, patient_id_to_text_all





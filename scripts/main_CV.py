# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:29:25 2025

@author: joelb
"""

import torch
import sys
import os
#os.chdir("C:/Users/joelb/OneDrive/Escritorio/4t/TFG/scripts/")
from data_utils.data_loader import get_fold_data
from data_utils.data_loader import load_multi_hospital_data
from training.train_CrossEntropy import train_retrieval
from evaluation.evaluate import confusion_matrix_by_field, extract_fields, visualize_attention_map
from models.encoders_baseline import CLIPMedical
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from collections import defaultdict
import glob
import seaborn as sns
from transformers import AutoTokenizer


plt.rcParams.update({
    "font.size": 25,            # tamaño base del texto
    "axes.titlesize": 24,       # títulos de los ejes
    "axes.labelsize": 24,       # etiquetas de los ejes
    "xtick.labelsize": 24,      # ticks del eje x
    "ytick.labelsize": 24,      # ticks del eje y
    "legend.fontsize": 24       # tamaño de la leyenda
})
# Para seaborn escalamos todo al mismo nivel
sns.set(font_scale=1.4  )
# --- Configuración ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DataDir = r'C:\Users\joelb\OneDrive\Escritorio\4t\TFG\data\CanRuti'
#npz_file_path_glcm = os.path.join(DataDir, 'ResNet152_slice_GLCM3D.npz')
#npz_file_path_intensity = os.path.join(DataDir, 'ResNet152_slice_intensity.npz')
#excel_meta_path = os.path.join(DataDir, '11112024_BDMetaData_CanRuti_RadDescriptors (1).xlsx')

# --- Configuración ---
k_folds = 2
batch_size = 32
num_epochs = 4

# --- Inicializar tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

# --- Directorios de hospitales ---
hospital_dirs = [
    r"C:\Users\joelb\OneDrive\Escritorio\4t\TFG\data\CanRuti",
    r"C:\Users\joelb\OneDrive\Escritorio\4t\TFG\data\DelMar",
    r"C:\Users\joelb\OneDrive\Escritorio\4t\TFG\data\MutuaTerrassa"
]

# --- Cargar datos combinados de múltiples hospitales ---
patient_id_to_image, patient_id_to_text = load_multi_hospital_data(hospital_dirs, tokenizer)

# --- Loop de validación cruzada ---
all_fold_metrics = []

for current_fold in range(k_folds):
    print(f"\n🔁 Fold {current_fold + 1}/{k_folds}")

    # --- Preparar datos para este fold ---
    train_loader, test_loader, train_ids, test_ids = get_fold_data(
        patient_id_to_image,  # Pasa los datos de imagen
        patient_id_to_text,   # Pasa los datos de texto
        k_folds=k_folds,
        current_fold=current_fold,
        batch_size=batch_size
    )

    # --- Inicializar modelo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPMedical().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- Entrenamiento ---
    train_retrieval(model, train_loader, optimizer, device, epochs=num_epochs)

    # --- Inferencia en test: construir catálogo y evaluar ---
    with torch.no_grad():
        test_image_features = [feat for feat, _, _ in test_loader.dataset]
        test_patient_ids = [pid for _, _, pid in test_loader.dataset]

        raw_train_texts = []
        for pid in train_ids:
            tokens = patient_id_to_text[pid]
            text_decoded = tokenizer.decode(tokens["input_ids"].squeeze(), skip_special_tokens=True)
            print(text_decoded)
            raw_train_texts.append(text_decoded)

        tokenized_catalog = tokenizer(
            raw_train_texts,
            padding='max_length', truncation=True, max_length=256, return_tensors="pt"
        )
        tokenized_catalog = {k: v.to(device) for k, v in tokenized_catalog.items()}
        catalog_embeddings = model.text_encoder(tokenized_catalog).detach().cpu()
        catalog_embeddings = catalog_embeddings.mean(dim=1)

        # Crear diccionario real de textos de test
        real_texts_dict = {
            pid: tokenizer.decode(patient_id_to_text[pid]["input_ids"].squeeze(), skip_special_tokens=True)
            for pid in test_patient_ids
        }

    # --- Evaluación en TEST ---
    top_k = 5
    top_k_texts_predicted = []
    for img_feat in test_image_features:
        img_tensor = img_feat.unsqueeze(0).to(device) if isinstance(img_feat, torch.Tensor) else torch.tensor(img_feat, dtype=torch.float32).unsqueeze(0).to(device)
        img_emb = model.image_projection(img_tensor).detach().cpu().numpy()

        if img_emb.ndim == 1:
            img_emb = img_emb.reshape(1, -1)

        sims = cosine_similarity(img_emb, catalog_embeddings.numpy())
        top_idxs = sims[0].argsort()[::-1][:top_k] # index dels textos mes similars
        top_sims = [sims[0][i] for i in top_idxs]
        top_texts = [raw_train_texts[i] for i in top_idxs]
        top_k_texts_predicted.append(top_texts)
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_k), top_sims, tick_label=[f'Texto {i+1}' for i in range(top_k)], color='purple')
        plt.title(f'Top {top_k} Most Similar Texts to the Image')
        plt.xlabel('Text')
        plt.ylabel('Cosine Similarity')
        plt.show()"""

    # --- Confusion Matrices por variable clínica ---
    confusion_matrix_by_field(
        test_ids=test_ids,
        real_texts_dict=real_texts_dict,
        top_k_texts_predicted=top_k_texts_predicted
    )

        # --- Reporte de métricas por campo ---
    metrics_by_field = defaultdict(list)
    
    for field in ["nodule shape", "nodule density","infiltration","cdiff","necrosis"]:
        y_true, y_pred = [], []
        
        for i, pid in enumerate(test_ids):
            real_text = real_texts_dict[pid]
            pred_text = top_k_texts_predicted[i][0]  # Top-1
    
            real_fields = extract_fields(real_text)
            pred_fields = extract_fields(pred_text)
    
            y_true.append(real_fields[field])
            y_pred.append(pred_fields[field])
    
        print(f"\n📊 Clasification Report for field: {field.upper()}")
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        
        for label, stats in report.items():
            if isinstance(stats, dict):  # Excluir 'accuracy'
                metrics_by_field[field].append({
                    "label": label,
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "f1-score": stats["f1-score"],
                    "support": stats["support"]
                })
    
    # Almacenar las métricas globales por fold si quieres después exportarlas
    print("\n📈 Métricas por fold recopiladas correctamente.")
    
    paciente_aciertos = []

    for i, pid in enumerate(test_ids):
        real_text = real_texts_dict[pid]
        pred_text = top_k_texts_predicted[i][0]  # top-1
    
        real_fields = extract_fields(real_text)
        pred_fields = extract_fields(pred_text)
    
        total_campos = 0
        campos_correctos = 0
    
        for field in ["nodule shape", "nodule density", "infiltration", "cdiff", "necrosis"]:
            val_real = real_fields.get(field)
            val_pred = pred_fields.get(field)
    
            # Ignora campos no definidos (por ejemplo, si hay NaN)
            if pd.notna(val_real) and pd.notna(val_pred):
                total_campos += 1
                if val_real == val_pred:
                    campos_correctos += 1
    
        if total_campos > 0:
            acierto_pct = campos_correctos / total_campos
            paciente_aciertos.append({
                "patient_id": pid,
                "correct_fields": campos_correctos,
                "total_fields": total_campos,
                "accuracy_pct": acierto_pct
            })
    
    # --- Análisis ---
    df_paciente = pd.DataFrame(paciente_aciertos)
    
    ############### TABLA RESUMEN + CASOS DE ESTUDIO ###################

    # Clasificar pacientes según su precisión
    df_paciente['classification'] = pd.cut(df_paciente['accuracy_pct'],
                                           bins=[-0.1, 0.2, 0.8, 1.1],
                                           labels=['≤20%', '≥80%', '100%'])
    
    # Tabla resumen con los contadores de pacientes por categoría
    summary_table = df_paciente['classification'].value_counts().sort_index()
    print("🔎 Summary of patient classifications:")
    print(summary_table)
    
    # Seleccionar un caso de estudio por cada grupo
    cases_100 = df_paciente[df_paciente['classification'] == '100%'].head(1)  # Caso con 100% de precisión
    cases_80 = df_paciente[df_paciente['classification'] == '≥80%'].head(1)  # Caso con ≥80% de precisión
    cases_20 = df_paciente[df_paciente['classification'] == '≤20%'].head(1)  # Caso con ≤20% de precisión
    
    print("\n📊 Cases for detailed analysis:")
    
    print("\n- Case with 100% accuracy:")
    print(cases_100)
    
    print("\n- Case with ≥80% accuracy:")
    print(cases_80)
    
    print("\n- Case with ≤20% accuracy:")
    print(cases_20)
    
    # Filtrar pacientes con precisión ≥ 80%
    patients_above_80 = df_paciente[df_paciente['accuracy_pct'] >= 0.8]
    
    print("\n🔍 Patients with ≥80% accuracy - Real vs Predicted texts:")
    
    # Imprimir los textos reales y predichos para estos pacientes
    for i, pid in patients_above_80.iterrows():
        patient_id = pid['patient_id']
        real_text = real_texts_dict[patient_id]  # Real text from the dataset
        pred_text = top_k_texts_predicted[patients_above_80.index.get_loc(i)][0]  # Predicted text (top-1)
    
        # Evaluar solo si el campo real no es 'nan'
        real_fields = extract_fields(real_text)
        pred_fields = extract_fields(pred_text)
    
        total_campos = 0
        campos_correctos = 0
    
        # Evaluación de campos de descriptores
        for field in ["nodule shape", "nodule density", "vinfiltration", "cdiff", "necrosis"]:
            val_real = real_fields.get(field, "missing")
            val_pred = pred_fields.get(field, "missing")
    
            # Contar como correcto solo si el valor real es no 'missing' o 'nan' y el predicho es igual
            if val_real != "missing" and pd.notna(val_real):
                total_campos += 1
                if val_real == val_pred:
                    campos_correctos += 1
    
        # Calcular el porcentaje de aciertos basados solo en campos válidos
        if total_campos > 0:
            accuracy_pct = campos_correctos / total_campos
            print(f"\nPatient ID: {patient_id}")
            print(f"Real Text: {real_text}")
            print(f"Predicted Text: {pred_text}")
            print(f"Accuracy on valid fields: {accuracy_pct * 100:.2f}%")
            print("-" * 80)  # Separator for readability
    
    ###################### HISTOGRAMA ##################
    
    # Histograma de pacientes por nivel de exactitud
    histograma = df_paciente["accuracy_pct"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.histplot(df_paciente["accuracy_pct"], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], discrete=False, edgecolor='black')
    
    plt.title("Distribución de precisión por paciente (descriptores clínicos)")
    plt.xlabel("Porcentaje de descriptores correctamente predichos")
    plt.ylabel("Número de pacientes")
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [f"{int(x*100)}%" for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    plt.grid(True)
    
    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig(f"results/ResNet152_final/patient_level_accuracy_histogram_fold_{current_fold + 1}.png")
    plt.show()
    
    print("\n📊 Distribución de precisión por paciente:")
    for pct, count in histograma.items():
        print(f"- {int(pct * 100)}% exactos: {count} pacientes")
    
    # Guardar CSV opcional
    df_paciente.to_csv(f"results/ResNet152_final/patient_level_accuracy_fold_{current_fold + 1}.csv", index=False)
    
    
    fold_metrics = []

    for field, stats_list in metrics_by_field.items():
        for stats in stats_list:
            fold_metrics.append({
                "fold": current_fold + 1,
                "field": field,
                "class": stats["label"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1_score": stats["f1-score"],
                "support": stats["support"]
            })
    
    df_metrics = pd.DataFrame(fold_metrics)
    
    # Crear carpeta si no existe
    os.makedirs("results/MobileNetV2_final/metrics_by_fold", exist_ok=True)
    
    # Guardar CSV del fold actual
    df_metrics.to_csv(f"results/MobileNetV2_final/metrics_by_fold/metrics_fold_{current_fold + 1}.csv", index=False)

print("\n📊 Promedio de métricas por clase y variable clínica:")

for field, stats_list in metrics_by_field.items():
    print(f"\n--- Field: {field.upper()} ---")
    df = pd.DataFrame(stats_list)
    grouped = df.groupby("label").agg(['mean', 'std'])
    print(grouped[["precision", "recall", "f1-score"]].round(3))
    
    
all_metrics = []

for fold in range(1, k_folds + 1):
    path = f"results/MobileNetV2_final/metrics_by_fold/metrics_fold_{fold}.csv"
    fold_df = pd.read_csv(path)
    all_metrics.append(fold_df)

global_df = pd.concat(all_metrics)

# Calcular medias y std por clase y campo
summary_df = global_df.groupby(["field", "class"]).agg({
    "precision": ["mean", "std"],
    "recall": ["mean", "std"],
    "f1_score": ["mean", "std"]
}).reset_index()

# Aplanar columnas jerárquicas
summary_df.columns = ['_'.join(col).strip("_") for col in summary_df.columns.values]

# Guardar CSV resumen
summary_df.to_csv("results/MobileNetV2_final/summary_metrics_across_folds.csv", index=False)
print("\n✅ CSV de resumen guardado como 'summary_metrics_across_folds.csv'")

all_csvs = glob.glob("results/MobileNetV2_final/metrics_by_fold/metrics_fold_*.csv")
df_all_metrics = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)

# Solo analizamos clases que no sean 'accuracy' o 'macro avg'
df_all_metrics = df_all_metrics[~df_all_metrics['class'].isin(['accuracy', 'macro avg', 'weighted avg'])]

# Crear carpeta para guardar los plots
os.makedirs("results/MobileNetV2_final/plots", exist_ok=True)

# Métricas que queremos graficar
metricas = ["precision", "recall", "f1_score"]

# Generar gráfico para cada métrica
for metrica in metricas:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="field", y=metrica, data=df_all_metrics)
    plt.title(f"Distribución de {metrica.upper()} por campo clínico (Cross-Validation)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.ylabel(metrica.upper(), fontsize=20)
    plt.xlabel("Campo clínico", fontsize=20)
    plt.xticks(fontsize=24)  # Ajustar tamaño de los números en el eje x
    plt.yticks(fontsize=24)
    plt.ylabel(metrica.upper())
    plt.xlabel("Campo clínico")
    plt.tight_layout()
    plt.savefig(f"results/MobileNetV2_final/plots/{metrica}_by_field_boxplot.png")
    plt.close()

print("✅ Gráficos de métricas por campo guardados en 'results/MobileNetV2_final/plots/'")

# Crear carpeta si no existe
os.makedirs("results/MobileNetV2_final/summary_by_metric", exist_ok=True)

# Agrupamos por campo clínico y calculamos media y std globales
descriptor_summary = global_df.groupby("field").agg({
    "precision": ["mean", "std"],
    "recall": ["mean", "std"],
    "f1_score": ["mean", "std"]
})

# Generar un CSV por cada métrica
for metric in ["precision", "recall", "f1_score"]:
    mean_series = descriptor_summary[(metric, "mean")].round(3)
    std_series = descriptor_summary[(metric, "std")].round(3)

    metric_data = {}
    for field in descriptor_summary.index:
        col_name = field.replace(" ", "_")
        metric_data[col_name] = [f"{mean_series[field]} ± {std_series[field]}"]

    df_metric = pd.DataFrame(metric_data)
    df_metric.to_csv(f"results/MobileNetV2_final/summary_by_metric/{metric}_summary.csv", index=False)
    print(f"✅ CSV guardado: {metric}_summary.csv")





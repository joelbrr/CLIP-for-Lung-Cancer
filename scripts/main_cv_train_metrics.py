# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:29:25 2025

@author: joelb
"""

import torch
import os
from data_utils.data_loader_attention import get_fold_data, load_multi_hospital_data, extract_patient_id
from training.train_CrossEntropy import train_retrieval
from training.train import train_model
from evaluation.evaluate import extract_fields
from models.encoders_baseline import CLIPMedical
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import seaborn as sns
from transformers import AutoTokenizer

plt.rcParams.update({
    "font.size": 25,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24
})
sns.set(font_scale=1.4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_folds = 2
batch_size = 32
num_epochs = 25

tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
hospital_dirs = [
    r"C:\\Users\\joelb\\OneDrive\\Escritorio\\4t\\TFG\\data\\CanRuti",
    r"C:\\Users\\joelb\\OneDrive\\Escritorio\\4t\\TFG\\data\\DelMar",
    r"C:\\Users\\joelb\\OneDrive\\Escritorio\\4t\\TFG\\data\\MutuaTerrassa"
]

patient_id_to_image, patient_id_to_text = load_multi_hospital_data(hospital_dirs, tokenizer, use_attention = True)
descriptor_fields = ["nodule shape", "nodule density", "infiltration", "cdiff", "necrosis"]
topk_values = [1, 3, 5]
all_topk_results = {k: [] for k in topk_values}

for current_fold in range(k_folds):
    print(f"\n🔁 Fold {current_fold + 1}/{k_folds}")

    train_loader, test_loader, train_ids, test_ids = get_fold_data(
        patient_id_to_image, patient_id_to_text, k_folds=k_folds,
        current_fold=current_fold, batch_size=batch_size
    )

    model = CLIPMedical().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_model(model, train_loader, optimizer, device, epochs=num_epochs, tokenizer = tokenizer)

    with torch.no_grad():
        raw_train_texts = [
            tokenizer.decode(patient_id_to_text[pid]["input_ids"].squeeze(), skip_special_tokens=True)
            for pid in train_ids
        ]
        tokenized_catalog = tokenizer(
            raw_train_texts, padding='max_length', truncation=True, max_length=256, return_tensors="pt"
        )
        tokenized_catalog = {k: v.to(device) for k, v in tokenized_catalog.items()}
        catalog_embeddings = model.text_encoder(tokenized_catalog).detach().cpu().mean(dim=1)

        test_features = [feat for feat, _, _ in test_loader.dataset]
        test_ids_list = [pid for _, _, pid in test_loader.dataset]
        topk_predictions = []

        for feat in test_features:
            tensor = feat.unsqueeze(0).to(device) if isinstance(feat, torch.Tensor) else torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            emb = model.image_projection(tensor).detach().cpu().numpy()
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            sims = cosine_similarity(emb, catalog_embeddings.numpy())
            top_idxs = sims[0].argsort()[::-1][:max(topk_values)]
            top_texts = [raw_train_texts[i] for i in top_idxs]
            topk_predictions.append(top_texts)

        real_texts_dict = {
            pid: tokenizer.decode(patient_id_to_text[pid]["input_ids"].squeeze(), skip_special_tokens=True)
            for pid in test_ids_list
        }

        for k in topk_values:
            print(f"\n🔍 Evaluando Top-{k}")
            metrics_by_field = defaultdict(list)

            for field in descriptor_fields:
                total = 0
                correct = 0
                for i, pid in enumerate(test_ids_list):
                    real_text = real_texts_dict[pid]
                    real_fields = extract_fields(real_text)
                    val_real = real_fields.get(field)
                    if pd.isna(val_real):
                        continue

                    pred_fields_k = [extract_fields(txt) for txt in topk_predictions[i][:k]]
                    pred_values_k = [pred.get(field) for pred in pred_fields_k if pred.get(field) is not None]

                    total += 1
                    if val_real in pred_values_k:
                        correct += 1

                accuracy = correct / total if total > 0 else 0.0
                metrics_by_field[field].append({
                    "fold": current_fold + 1,
                    "top_k": k,
                    "field": field,
                    "accuracy": accuracy
                })

            for field, values in metrics_by_field.items():
                all_topk_results[k].extend(values)

# Guardar resultados por top-k y graficar
output_dir = "results/attention_mechanism/topk_analysis"
os.makedirs(output_dir, exist_ok=True)
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

all_data = []
for k in topk_values:
    df_k = pd.DataFrame(all_topk_results[k])
    df_k.to_csv(f"{output_dir}/metrics_top{k}.csv", index=False)
    df_k["top_k"] = f"Top-{k}"
    all_data.append(df_k)

    summary = df_k.groupby("field").agg({"accuracy": ["mean", "std"]})
    mean_series = summary[("accuracy", "mean")].round(3)
    std_series = summary[("accuracy", "std")].round(3)
    data = {field.replace(" ", "_"): [f"{mean_series[field]} ± {std_series[field]}"] for field in summary.index}
    df_metric = pd.DataFrame(data)
    df_metric.to_csv(f"{output_dir}/accuracy_top{k}_summary.csv", index=False)
    print(f"✅ CSV guardado: accuracy_top{k}_summary.csv")

# Crear gráficas comparativas
combined_df = pd.concat(all_data, ignore_index=True)

plt.figure(figsize=(12, 6))
sns.barplot(x="field", y="accuracy", hue="top_k", data=combined_df, ci="sd")
plt.title("Comparación de ACCURACY por descriptor y top-k")
plt.ylim(0, 1)
plt.ylabel("ACCURACY")
plt.xlabel("Descriptor clínico")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "comparison_accuracy.png"))
plt.close()


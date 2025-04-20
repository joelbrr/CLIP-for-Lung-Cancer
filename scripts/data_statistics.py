# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:38:20 2025

@author: joelb
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2_contingency

# --- Variables que queremos estudiar ---
CLINICAL_FIELDS = ["nodule_shape", "nodule_density", "vinfiltration", "cdiff", "necrosis"]

def load_all_metadata(hospital_dirs):
    dfs = []
    for path in hospital_dirs:
        for file in os.listdir(path):
            if file.endswith(".xlsx") and "MetaData" in file:
                full_path = os.path.join(path, file)
                df = pd.read_excel(full_path)
                df["hospital"] = os.path.basename(path)
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def describe_field_distribution(df, output_dir="results/statistics"):
    os.makedirs(output_dir, exist_ok=True)
    for field in CLINICAL_FIELDS:
        if field in df.columns:
            counts = df[field].value_counts(dropna=False).reset_index()
            counts.columns = [field, "count"]
            counts.to_csv(os.path.join(output_dir, f"distribution_{field}.csv"), index=False)
            # Graficar
            plt.figure(figsize=(8, 4))
            sns.barplot(data=counts, x=field, y="count")
            plt.title(f"Distribución de {field}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"plot_distribution_{field}.png"))
            plt.close()
            print(f"Guardado: distribution_{field}.csv y gráfico")
        else:
            print(f"Campo no encontrado en el dataset: {field}")

def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def analyze_field_associations(df, output_dir="results/statistics"):
    os.makedirs(output_dir, exist_ok=True)
    associations = pd.DataFrame(index=CLINICAL_FIELDS, columns=CLINICAL_FIELDS)

    for i, var1 in enumerate(CLINICAL_FIELDS):
        for j, var2 in enumerate(CLINICAL_FIELDS):
            if var1 != var2:
                ct = pd.crosstab(df[var1], df[var2])
                if ct.shape[0] > 1 and ct.shape[1] > 1:
                    chi2, p, _, _ = stats.chi2_contingency(ct)
                    v = cramers_v(ct)
                    associations.loc[var1, var2] = round(v, 3)

    # Guardar heatmap de Cramer's V
    plt.figure(figsize=(10, 8))
    sns.heatmap(associations.astype(float), annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Cramér's V entre variables clínicas")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cramers_v_heatmap.png"))
    plt.close()
    print("✅ Heatmap de Cramér's V guardado.")
    
    
def analyze_chi2_correlation(df, output_dir="results/statistics/chi2"):
    """
    Analiza la dependencia entre pares de variables categóricas usando el test Chi-cuadrado.
    """
    os.makedirs(output_dir, exist_ok=True)
    heatmap_matrix = pd.DataFrame(index=CLINICAL_FIELDS, columns=CLINICAL_FIELDS)

    for i, field1 in enumerate(CLINICAL_FIELDS):
        for j, field2 in enumerate(CLINICAL_FIELDS):
            if i >= j:
                heatmap_matrix.loc[field1, field2] = None
                continue

            # Tabla de contingencia
            contingency = pd.crosstab(df[field1], df[field2])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                heatmap_matrix.loc[field1, field2] = 0
                continue

            chi2, p, _, _ = chi2_contingency(contingency)
            heatmap_matrix.loc[field1, field2] = chi2

    # Graficar heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_matrix.astype(float), annot=True, cmap="YlGnBu")
    plt.title("Chi² statistic entre pares de variables clínicas")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chi2_heatmap.png"))
    plt.show()
    print("✅ Heatmap de Chi² guardado.")


def analyze_cooccurrence_patterns(df, output_dir="results/statistics/cooccurrence"):
    """
    Analiza qué combinaciones de valores clínicos aparecen con mayor frecuencia.
    """
    os.makedirs(output_dir, exist_ok=True)
    combo_counts = df[CLINICAL_FIELDS].fillna("missing").value_counts().reset_index()
    combo_counts.columns = CLINICAL_FIELDS + ["count"]
    combo_counts.to_csv(os.path.join(output_dir, "cooccurrence_combinations.csv"), index=False)

    # Mostrar top 10 combinaciones
    print("\n🔁 Top combinaciones clínicas más frecuentes:")
    print(combo_counts.head(10).to_string(index=False))

if __name__ == "__main__":
    hospital_dirs = [
        r"C:/Users/joelb/OneDrive/Escritorio/4t/TFG/data/CanRuti",
        r"C:/Users/joelb/OneDrive/Escritorio/4t/TFG/data/DelMar",
        r"C:/Users/joelb/OneDrive/Escritorio/4t/TFG/data/MutuaTerrassa"
    ]

    df_all = load_all_metadata(hospital_dirs)
    describe_field_distribution(df_all)
    analyze_field_associations(df_all)
    analyze_chi2_correlation(df_all)
    analyze_cooccurrence_patterns(df_all)
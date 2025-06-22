# -*- coding: utf-8 -*-
"""
Actualizado el 25‑May‑2025

Cambios principales
-------------------
1. **Fuentes más grandes y legibles** en todas las gráficas:
   * Se establece un `font_scale` global con *Seaborn* y se ajustan tamaños
     de títulos, ejes y anotaciones.
   * Los heatmaps usan `annot_kws` para que los números se vean claros.
2. **Tabla de distribuciones clínicas agregada** (`distribution_summary_table.csv`).
   * Cada fila = variable clínica.
   * Cada columna = categoría con el número de pacientes.

Uso:
-----
Ejecutar directamente este script (`python data_statistics_updated.py`) o
importar las funciones desde otro cuaderno.
"""

import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2_contingency

# ───────────────────────────────────────────────────────────────────────────────
# Configuración global de estilo (fuentes grandes y legibles)
# ───────────────────────────────────────────────────────────────────────────────

sns.set_theme(context="notebook", style="whitegrid", font_scale=1.8)
plt.rcParams.update({
    "figure.autolayout": True,   # evita que los textos se recorten
    "axes.titlesize": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# --- Variables que queremos estudiar ---
CLINICAL_FIELDS = [
    "nodule_shape",
    "nodule_density",
    "vinfiltration",
    "cdiff",
    "necrosis",
]

# ───────────────────────────────────────────────────────────────────────────────
# Carga de metadatos desde distintos hospitales
# ───────────────────────────────────────────────────────────────────────────────

def load_all_metadata(hospital_dirs: List[str]) -> pd.DataFrame:
    """Carga y concatena los archivos *MetaData*.xlsx de múltiples carpetas."""
    dfs = []
    for path in hospital_dirs:
        for file in os.listdir(path):
            if file.endswith(".xlsx") and "MetaData" in file:
                full_path = os.path.join(path, file)
                df = pd.read_excel(full_path)
                df["hospital"] = os.path.basename(path)
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ───────────────────────────────────────────────────────────────────────────────
# Distribución individual de cada campo clínico (csv + gráfico)
# ───────────────────────────────────────────────────────────────────────────────

def describe_field_distribution(df: pd.DataFrame, output_dir: str = "results/statistics") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for field in CLINICAL_FIELDS:
        if field not in df.columns:
            print(f"⚠️  Campo no encontrado en el dataset: {field}")
            continue

        df_field = df[field].fillna("missing").astype(str)
        counts = df_field.value_counts().reset_index()
        counts.columns = [field, "count"]
        counts.to_csv(os.path.join(output_dir, f"distribution_{field}.csv"), index=False)

        # ── Gráfica ────────────────────────────────────────────────────────────
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=counts, x=field, y="count", color="#4C72B0")
        ax.set_title(f"Distribución de {field}")
        ax.set_xlabel(field)
        ax.set_ylabel("Número de pacientes")
        plt.xticks(rotation=45, ha="right")

        # Pie de figura / fuente
        plt.figtext(
            0.99,
            0.01,
            ".",
            ha="right",
            va="bottom",
            fontsize=12,
            fontstyle="italic",
        )

        plt.savefig(os.path.join(output_dir, f"plot_distribution_{field}.png"), dpi=150)
        plt.close()
        print(f"✅ Guardado distribution_{field}.csv y su gráfico.")

# ───────────────────────────────────────────────────────────────────────────────
# Tabla resumen de distribuciones (una fila = variable clínica)
# ───────────────────────────────────────────────────────────────────────────────
def create_distribution_summary_table(
    df: pd.DataFrame,
    clinical_fields: List[str] = CLINICAL_FIELDS,
    output_dir: str = "results/statistics",
) -> pd.DataFrame:
    """Genera un CSV donde cada fila es una variable clínica y cada columna
    una categoría con su conteo de pacientes."""

    os.makedirs(output_dir, exist_ok=True)

    # Construir un dict de dicts: {variable→{valor:str → n}}
    summary_records = {}
    for field in clinical_fields:
        if field in df.columns:
            counts = (
                df[field]
                .fillna("NaN")  # cuenta los NaN como categoría
                .astype(str)
                .value_counts()
            )
            summary_records[field] = counts.to_dict()

    # Pasar a DataFrame sin alinear índices entre variables
    summary_df = (
        pd.DataFrame.from_dict(summary_records, orient="index")
        .fillna(0)
        .astype(int)
    )

    # Ordenar columnas alfabéticamente (todas son str)
    summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)

    csv_path = os.path.join(output_dir, "distribution_summary_table.csv")
    summary_df.to_csv(csv_path)
    print(f"✅ Tabla de distribuciones guardada en {csv_path}")

    return summary_df

# ───────────────────────────────────────────────────────────────────────────────
# Asociación entre pares de variables clínicas (Cramér's V)
# ───────────────────────────────────────────────────────────────────────────────

def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / max((kcorr - 1), (rcorr - 1)))

def analyze_field_associations(df: pd.DataFrame, output_dir: str = "results/statistics") -> None:
    os.makedirs(output_dir, exist_ok=True)
    associations = pd.DataFrame(index=CLINICAL_FIELDS, columns=CLINICAL_FIELDS)

    for var1 in CLINICAL_FIELDS:
        for var2 in CLINICAL_FIELDS:
            if var1 == var2:
                associations.loc[var1, var2] = 1.0  # perfecta concordancia consigo misma
                continue
            ct = pd.crosstab(df[var1], df[var2])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                associations.loc[var1, var2] = round(cramers_v(ct), 3)
            else:
                associations.loc[var1, var2] = 0

    # ── Heatmap ────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        associations.astype(float),
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        annot_kws={"size": 20},
    )
    plt.title("Cramér's V entre variables clínicas")

    plt.figtext(
        0.99,
        0.01,
        ".",
        ha="right",
        va="bottom",
        fontsize=12,
        fontstyle="italic",
    )

    plt.savefig(os.path.join(output_dir, "cramers_v_heatmap.png"), dpi=150)
    plt.close()
    print("✅ Heatmap de Cramér's V guardado.")

# ───────────────────────────────────────────────────────────────────────────────
# Chi‑cuadrado entre pares de variables categóricas
# ───────────────────────────────────────────────────────────────────────────────

def analyze_chi2_correlation(df: pd.DataFrame, output_dir: str = "results/statistics/chi2") -> None:
    os.makedirs(output_dir, exist_ok=True)
    heatmap_matrix = pd.DataFrame(index=CLINICAL_FIELDS, columns=CLINICAL_FIELDS)

    for i, field1 in enumerate(CLINICAL_FIELDS):
        for j, field2 in enumerate(CLINICAL_FIELDS):
            if i >= j:
                heatmap_matrix.loc[field1, field2] = np.nan
                continue
            contingency = pd.crosstab(df[field1], df[field2])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                heatmap_matrix.loc[field1, field2] = 0
                continue
            chi2, _, _, _ = chi2_contingency(contingency)
            heatmap_matrix.loc[field1, field2] = chi2

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_matrix.astype(float),
        annot=True,
        cmap="YlGnBu",
        annot_kws={"size": 20},
    )
    plt.title("Chi² entre pares de variables clínicas")

    plt.figtext(
        0.99,
        0.01,
        ".",
        ha="right",
        va="bottom",
        fontsize=14,
        fontstyle="italic",
    )

    plt.savefig(os.path.join(output_dir, "chi2_heatmap.png"), dpi=150)
    plt.close()
    print("✅ Heatmap de Chi² guardado.")

# ───────────────────────────────────────────────────────────────────────────────
# Análisis de co‑ocurrencias frecuentes
# ───────────────────────────────────────────────────────────────────────────────

def analyze_cooccurrence_patterns(df: pd.DataFrame, output_dir: str = "results/statistics/cooccurrence") -> None:
    os.makedirs(output_dir, exist_ok=True)
    combo_counts = df[CLINICAL_FIELDS].fillna("missing").value_counts().reset_index()
    combo_counts.columns = CLINICAL_FIELDS + ["count"]
    csv_path = os.path.join(output_dir, "cooccurrence_combinations.csv")
    combo_counts.to_csv(csv_path, index=False)

    print("\n🔁 Top 10 combinaciones clínicas más frecuentes:")
    print(combo_counts.head(10).to_string(index=False))

    print(f"✅ Combinaciones completas guardadas en {csv_path}")

# ───────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hospital_dirs = [
        r"C:/Users/joelb/OneDrive/Escritorio/4t/TFG/data/CanRuti",
        r"C:/Users/joelb/OneDrive/Escritorio/4t/TFG/data/DelMar",
        r"C:/Users/joelb/OneDrive/Escritorio/4t/TFG/data/MutuaTerrassa",
    ]

    df_all = load_all_metadata(hospital_dirs)

    # Distribuciones individuales + gráficas
    describe_field_distribution(df_all)

    # Tabla de distribuciones completa
    create_distribution_summary_table(df_all,CLINICAL_FIELDS)

    # Asociación de variables clínicas
    analyze_field_associations(df_all)
    analyze_chi2_correlation(df_all)

    # Co‑ocurrencias
    analyze_cooccurrence_patterns(df_all)


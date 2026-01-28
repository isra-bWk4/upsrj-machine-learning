import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer archivo
SOURCE_FILE = os.path.join(".", "inputs", "rutina.0.csv")
df = pd.read_csv(SOURCE_FILE)

# Convertir semanas a numéricas
semanas = ["Semana 1","Semana 2","Semana 3","Semana 4","Semana 5"]
for semana in semanas:
    if semana in df.columns:
        df[semana] = pd.to_numeric(df[semana], errors="coerce")

# Clasificación de progreso según repeticiones
def clasificar(valor):
    if pd.isna(valor):
        return "Sin datos"
    elif valor <= 13:
        return "Lejos de la meta"
    elif 14 <= valor <= 17:
        return "Cerca de la meta"
    elif 18 <= valor <= 19:
        return "Muy cerca de la meta"
    elif valor == 20:
        return "Meta alcanzada"
    else:
        return "Sin datos"

# Función para mostrar datos de una semana específica
def mostrar_semana(df, semana):
    print(f"\n--- {semana} ---")
    if df[semana].isnull().all():
        print("No hay datos aún.")
    else:
        # Calcular faltantes
        df[f"faltan_{semana}"] = df["Repeticiones sug."] - df[semana]

        # Estado y progreso
        df["estado"] = np.where(df[semana] >= df["Repeticiones sug."], "Completado", "Pendiente")
        df["progreso"] = df[semana].apply(clasificar)

        # Mostrar todo junto
        print(df[["Día","Ejercicio","Repeticiones sug.",semana,f"faltan_{semana}","estado","progreso"]])

# Función para comparar dos semanas con gráfica
def comparar_semanas(df, semana_a, semana_b):
    print(f"\n--- Comparación {semana_a} vs {semana_b} ---")
    if df[semana_a].isnull().all() or df[semana_b].isnull().all():
        print("No hay datos suficientes para comparar.")
    else:
        df[f"diferencia_{semana_b}_vs_{semana_a}"] = df[semana_b] - df[semana_a]

        # Estado y progreso basados en semana_b
        df["estado"] = np.where(df[semana_b] >= df["Repeticiones sug."], "Completado", "Pendiente")
        df["progreso"] = df[semana_b].apply(clasificar)

        print(df[["Día","Ejercicio",semana_a,semana_b,f"diferencia_{semana_b}_vs_{semana_a}","estado","progreso"]])

        # Gráfica de comparación
        plt.figure(figsize=(12,6))
        x = np.arange(len(df["Ejercicio"]))
        width = 0.35
        plt.bar(x - width/2, df[semana_a].fillna(0), width, label=semana_a)
        plt.bar(x + width/2, df[semana_b].fillna(0), width, label=semana_b)
        plt.xticks(x, df["Ejercicio"], rotation=90)
        plt.ylim(0,20)
        plt.ylabel("Repeticiones")
        plt.title(f"Comparación {semana_a} vs {semana_b}")
        plt.legend()
        plt.tight_layout()
        OUTPUT_FIG = os.path.join(".", "out", f"comparacion_{semana_a}_vs_{semana_b}.png")
        plt.savefig(OUTPUT_FIG)
        plt.close()   # cerramos la figura para evitar warnings

# -------------------------------
# Ejemplo: mostrar semana seleccionada
print("\n--- Semana seleccionada---")
mostrar_semana(df, "Semana 5")

# Ejemplo: comparación entre semanas
print("\n--- Comparación ---")
comparar_semanas(df, "Semana 3", "Semana 4")

# Comparación final: última semana vs sugeridas
ultima_semana = df[semanas].ffill(axis=1).iloc[:, -1]   # toma el último valor no nulo
df["ultima_semana"] = ultima_semana
df["diferencia_meta"] = df["Repeticiones sug."] - df["ultima_semana"]

# Estado y progreso finales
df["estado"] = np.where(df["ultima_semana"] >= df["Repeticiones sug."], "Completado", "Pendiente")
df["progreso"] = df["ultima_semana"].apply(clasificar)

print("\n--- Comparación final ---")
print(df[["Día","Ejercicio","Repeticiones sug.","ultima_semana","diferencia_meta","estado","progreso"]])

# Gráfica final: última semana vs sugeridas
plt.figure(figsize=(12,6))
x = np.arange(len(df["Ejercicio"]))
width = 0.35
plt.bar(x - width/2, df["ultima_semana"].fillna(0), width, label="Última semana")
plt.bar(x + width/2, df["Repeticiones sug."].fillna(0), width, label="Sugeridas")
plt.xticks(x, df["Ejercicio"], rotation=90)
plt.ylim(0,20)
plt.ylabel("Repeticiones")
plt.title("Comparación final: Última semana vs Repeticiones sugeridas")
plt.legend()
plt.tight_layout()
OUTPUT_FIG = os.path.join(".", "out", "comparacion_final.png")
plt.savefig(OUTPUT_FIG)
plt.close()   # cerramos la figura para evitar warnings
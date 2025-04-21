import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Example label maps (replace with your actual label maps)
label_to_cls_piezas = {
    1: "Antiniebla delantero derecho",
    2: "Antiniebla delantero izquierdo",
    3: "Capó",
    # ... add all labels as needed
}

label_to_cls_danos = {
    1: "Abolladura",
    2: "Deformación",
    3: "Desprendimiento",
    4: "Fractura",
    5: "Rayón",
    6: "Rotura"
}

label_to_cls_sugerencia = {
    1: "Reparar",
    2: "Reemplazar"
}

def plot_label_distribution(df, column, label_map, title):
    """
    Plot distribution of multi-label column.
    df: DataFrame containing the data
    column: column name with list of labels
    label_map: dict mapping label ids to names
    title: plot title
    """
    all_labels = []
    for labels_list in df[column]:
        all_labels.extend(labels_list)
    label_counts = pd.Series(all_labels).value_counts().sort_index()

    label_names = [label_map.get(i, f"Label {i}") for i in label_counts.index]

    plt.figure(figsize=(12,6))
    sns.barplot(x=label_names, y=label_counts.values, palette="viridis")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel("Cantidad de ocurrencias")
    plt.xlabel("Etiquetas")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_multilabel(y_true, y_pred, class_names):
    """
    Plot confusion matrix for each class in multi-label classification.
    y_true, y_pred: lists of lists of binary labels (0/1)
    class_names: list of class names
    """
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix([y[i] for y in y_true], [y[i] for y in y_pred])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
        plt.title(f"Matriz de Confusión para clase: {class_name}")
        plt.show()

# Example usage:
# Load your train_df DataFrame with multi-label columns 'partes', 'dannos', 'sugerencias'
# plot_label_distribution(train_df, 'partes', label_to_cls_piezas, "Distribución de Piezas del Vehículo en Entrenamiento")
# plot_label_distribution(train_df, 'dannos', label_to_cls_danos, "Distribución de Tipos de Daño en Entrenamiento")
# plot_label_distribution(train_df, 'sugerencias', label_to_cls_sugerencia, "Distribución de Sugerencias en Entrenamiento")

# For confusion matrix, you need y_true and y_pred in binary indicator format:
# y_true = [[1,0,1], [0,1,0], [1,1,0], [0,0,1]]
# y_pred = [[1,0,0], [0,1,1], [1,0,0], [0,0,1]]
# class_names = ['Clase 1', 'Clase 2', 'Clase 3']
# plot_confusion_matrix_multilabel(y_true, y_pred, class_names)

import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Label maps from your notebook
label_to_cls_piezas = {
    1: "Antiniebla delantero derecho",
    2: "Antiniebla delantero izquierdo",
    3: "Capó",
    4: "Cerradura capo",
    5: "Cerradura maletero",
    6: "Cerradura puerta",
    7: "Espejo lateral derecho",
    8: "Espejo lateral izquierdo",
    9: "Faros derecho",
    10: "Faros izquierdo",
    11: "Guardabarros delantero derecho",
    12: "Guardabarros delantero izquierdo",
    13: "Guardabarros trasero derecho",
    14: "Guardabarros trasero izquierdo",
    15: "Luz indicadora delantera derecha",
    16: "Luz indicadora delantera izquierda",
    17: "Luz indicadora trasera derecha",
    18: "Luz indicadora trasera izquierda",
    19: "Luz trasera derecho",
    20: "Luz trasera izquierdo",
    21: "Maletero",
    22: "Manija derecha",
    23: "Manija izquierda",
    24: "Marco de la ventana",
    25: "Marco de las puertas",
    26: "Moldura capó",
    27: "Moldura puerta delantera derecha",
    28: "Moldura puerta delantera izquierda",
    29: "Moldura puerta trasera derecha",
    30: "Moldura puerta trasera izquierda",
    31: "Parabrisas delantero",
    32: "Parabrisas trasero",
    33: "Parachoques delantero",
    34: "Parachoques trasero",
    35: "Puerta delantera derecha",
    36: "Puerta delantera izquierda",
    37: "Puerta trasera derecha",
    38: "Puerta trasera izquierda",
    39: "Rejilla, parrilla",
    40: "Rueda",
    41: "Tapa de combustible",
    42: "Tapa de rueda",
    43: "Techo",
    44: "Techo corredizo",
    45: "Ventana delantera derecha",
    46: "Ventana delantera izquierda",
    47: "Ventana trasera derecha",
    48: "Ventana trasera izquierda",
    49: "Ventanilla delantera derecha",
    50: "Ventanilla delantera izquierda",
    51: "Ventanilla trasera derecha",
    52: "Ventanilla trasera izquierda"
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

def convert_string_lists(df):
    for col in ['dannos', 'partes', 'sugerencias']:
        df[col] = df[col].apply(ast.literal_eval)
    return df

def plot_label_distribution(df, column, label_map, title):
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
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix([y[i] for y in y_true], [y[i] for y in y_pred])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
        plt.title(f"Matriz de Confusión para clase: {class_name}")
        plt.show()

# Load datasets
train_df = pd.read_csv('data/fotos_siniestros/datasets/multi_train.csv', sep='|')
val_df = pd.read_csv('data/fotos_siniestros/datasets/multi_val.csv', sep='|')
test_df = pd.read_csv('data/fotos_siniestros/datasets/multi_test.csv', sep='|')

# Convert string lists to actual lists
train_df = convert_string_lists(train_df)
val_df = convert_string_lists(val_df)
test_df = convert_string_lists(test_df)

# Plot label distributions for training set
plot_label_distribution(train_df, 'partes', label_to_cls_piezas, "Distribución de Piezas del Vehículo en Entrenamiento")
plot_label_distribution(train_df, 'dannos', label_to_cls_danos, "Distribución de Tipos de Daño en Entrenamiento")
plot_label_distribution(train_df, 'sugerencias', label_to_cls_sugerencia, "Distribución de Sugerencias en Entrenamiento")

# Example placeholder for confusion matrix plotting:
# You need to prepare y_true and y_pred as binary indicator lists for each class.
# For example:
# y_true = [[1,0,1], [0,1,0], ...]
# y_pred = [[1,0,0], [0,1,1], ...]
# class_names = list(label_to_cls_danos.values())  # or other label sets
# plot_confusion_matrix_multilabel(y_true, y_pred, class_names)

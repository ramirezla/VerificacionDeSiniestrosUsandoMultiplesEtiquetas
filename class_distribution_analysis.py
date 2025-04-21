import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df, column, label_map, title):
    """
    Plot bar and pie charts for class distribution in multi-label data.
    df: DataFrame with multi-label column (list of labels)
    column: column name
    label_map: dict mapping label ids to names
    title: plot title
    """
    # Flatten all labels
    all_labels = []
    for labels_list in df[column]:
        all_labels.extend(labels_list)
    label_counts = pd.Series(all_labels).value_counts().sort_index()
    label_names = [label_map.get(i, f"Label {i}") for i in label_counts.index]

    # Bar plot
    plt.figure(figsize=(14,6))
    sns.barplot(x=label_names, y=label_counts.values, palette="coolwarm")
    plt.xticks(rotation=90)
    plt.title(f"{title} - Conteo de Clases")
    plt.ylabel("Número de ocurrencias")
    plt.xlabel("Clases")
    plt.tight_layout()
    plt.show()

    # Pie chart
    plt.figure(figsize=(8,8))
    plt.pie(label_counts.values, labels=label_names, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("coolwarm", len(label_counts)))
    plt.title(f"{title} - Proporción de Clases")
    plt.axis('equal')
    plt.show()

    # Print imbalance info
    total = label_counts.sum()
    print(f"Total etiquetas en {title}: {total}")
    print("Distribución de clases (porcentaje):")
    for name, count in zip(label_names, label_counts.values):
        print(f"  {name}: {count} ({count/total:.2%})")

def exhaustive_class_distribution_analysis(train_df, label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencia):
    print("Análisis exhaustivo de distribución de clases para el conjunto de entrenamiento\n")

    plot_class_distribution(train_df, 'partes', label_to_cls_piezas, "Piezas del Vehículo")
    plot_class_distribution(train_df, 'dannos', label_to_cls_danos, "Tipos de Daño")
    plot_class_distribution(train_df, 'sugerencias', label_to_cls_sugerencia, "Sugerencias")

    print("\nConsideraciones para manejo de desbalanceo:")
    print("- Técnicas de sobremuestreo (SMOTE, ADASYN)")
    print("- Técnicas de submuestreo")
    print("- Ajuste de pesos en la función de pérdida")
    print("- Uso de métricas adecuadas para datos desbalanceados (F1-score, AUC-PR)")

# Ejemplo de uso:
# train_df debe tener las columnas 'partes', 'dannos', 'sugerencias' como listas de etiquetas
# exhaustive_class_distribution_analysis(train_df, label_to_cls_piezas, label_to_cls_danos, label_to_cls_sugerencia)

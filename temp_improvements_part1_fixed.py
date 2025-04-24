import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# Using smote-variants library for multi-label oversampling
import smote_variants as sv

def apply_oversampling(X, y, method='MLSMOTE'):
    """
    Apply multi-label oversampling using smote-variants.
    X: feature matrix
    y: multi-label binary matrix
    method: 'MLSMOTE' or 'MLADASYN'
    Returns: resampled X and y
    """
    if method == 'MLSMOTE':
        sampler = sv.ML_SMOTE()
    elif method == 'MLADASYN':
        sampler = sv.ML_ADASYN()
    else:
        raise ValueError("Unsupported method. Choose 'MLSMOTE' or 'MLADASYN'.")
    X_resampled, y_resampled = sampler.sample(X, y)
    return X_resampled, y_resampled

# Example usage with your data:

# Convert multi-label columns to binary indicator matrices
mlb_parts = MultiLabelBinarizer(classes=sorted(cls_to_label_piezas.values()))
mlb_dannos = MultiLabelBinarizer(classes=sorted(cls_to_label_danos.values()))
mlb_sugerencias = MultiLabelBinarizer(classes=sorted(cls_to_label_sugerencia.values()))

parts_binary = mlb_parts.fit_transform(multi_consolidado_encoded['partes'])
dannos_binary = mlb_dannos.fit_transform(multi_consolidado_encoded['dannos'])
sugerencias_binary = mlb_sugerencias.fit_transform(multi_consolidado_encoded['sugerencias'])

# Combine all labels into a single multi-label matrix for oversampling
combined_labels = np.hstack([parts_binary, dannos_binary, sugerencias_binary])

# For oversampling, we need features X. Since this is image data, features could be image embeddings.
# Here, as a placeholder, we use the index as features (not ideal, replace with actual features or embeddings)
X_placeholder = np.arange(len(multi_consolidado_encoded)).reshape(-1, 1)

# Apply oversampling using multi-label method
X_resampled, y_resampled = apply_oversampling(X_placeholder, combined_labels, method='MLSMOTE')

# After resampling, reconstruct the DataFrame with oversampled labels
num_parts = parts_binary.shape[1]
num_dannos = dannos_binary.shape[1]
num_sugerencias = sugerencias_binary.shape[1]

parts_resampled = y_resampled[:, :num_parts]
dannos_resampled = y_resampled[:, num_parts:num_parts+num_dannos]
sugerencias_resampled = y_resampled[:, num_parts+num_dannos:]

# Convert binary matrices back to label lists
def binary_to_label_list(binary_matrix, classes):
    label_lists = []
    for row in binary_matrix:
        labels = [classes[i] for i, val in enumerate(row) if val == 1]
        label_lists.append(labels)
    return label_lists

parts_labels_resampled = binary_to_label_list(parts_resampled, mlb_parts.classes_)
dannos_labels_resampled = binary_to_label_list(dannos_resampled, mlb_dannos.classes_)
sugerencias_labels_resampled = binary_to_label_list(sugerencias_resampled, mlb_sugerencias.classes_)

# Create oversampled DataFrame
df_oversampled = pd.DataFrame({
    'Imagen': [f"resampled_{i}" for i in range(len(X_resampled))],  # Placeholder image names
    'partes': parts_labels_resampled,
    'dannos': dannos_labels_resampled,
    'sugerencias': sugerencias_labels_resampled
})

print(f"Original dataset size: {len(multi_consolidado_encoded)}")
print(f"Oversampled dataset size: {len(df_oversampled)}")

# Proceed with train/val/test split on df_oversampled as needed

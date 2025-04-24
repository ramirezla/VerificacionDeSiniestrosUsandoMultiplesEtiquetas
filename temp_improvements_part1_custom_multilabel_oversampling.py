import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from imblearn.over_sampling import SMOTE

def apply_multilabel_oversampling(X, y, method='SMOTE', k_neighbors=3):
    """
    Apply oversampling to multi-label data by converting to single-label, oversampling, then converting back.
    X: feature matrix
    y: multi-label binary matrix
    method: 'SMOTE' or 'ADASYN'
    k_neighbors: parameter for SMOTE/ADASYN
    Returns: resampled X and y (multi-label)
    """
    from imblearn.over_sampling import SMOTE, ADASYN
    import numpy as np
    from collections import Counter

    y_single, label_to_int = multilabel_to_singlelabel(y)

    # Determine minimum samples in any class to adjust k_neighbors
    class_counts = Counter(y_single)
    min_class_samples = min(class_counts.values())
    if min_class_samples <= 1:
        raise ValueError(f"Cannot perform oversampling because the smallest class has {min_class_samples} sample(s).")
    adjusted_k = min(k_neighbors, min_class_samples - 1)

    if method == 'SMOTE':
        sampler = SMOTE(k_neighbors=adjusted_k, random_state=42)
    elif method == 'ADASYN':
        sampler = ADASYN(n_neighbors=adjusted_k, random_state=42)
    else:
        raise ValueError("Unsupported method. Choose 'SMOTE' or 'ADASYN'.")

    X_resampled, y_single_resampled = sampler.fit_resample(X, y_single)

    y_resampled = singlelabel_to_multilabel(y_single_resampled, label_to_int)

    return X_resampled, y_resampled

# Example usage:
# X_resampled, y_resampled = apply_multilabel_oversampling(X_placeholder, combined_labels, method='SMOTE', k_neighbors=3)

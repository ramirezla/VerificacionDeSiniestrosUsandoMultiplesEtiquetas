import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_multilabel_class_distribution(y, class_names=None):
    """
    Analyze and plot the distribution of classes in a multi-label binary matrix.
    y: numpy array of shape (n_samples, n_classes)
    class_names: list of class names corresponding to columns in y
    """
    # Sum occurrences of each class
    class_counts = np.sum(y, axis=0)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(y.shape[1])]

    # Create a sorted list of (class_name, count)
    sorted_counts = sorted(zip(class_names, class_counts), key=lambda x: x[1])

    # Print classes with low counts
    print("Classes with low sample counts:")
    for cls, count in sorted_counts:
        if count < 10:  # Threshold for rarity, adjust as needed
            print(f"{cls}: {count} samples")

    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar([cls for cls, _ in sorted_counts], [count for _, count in sorted_counts])
    plt.xticks(rotation=90)
    plt.title("Multi-label Class Distribution")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

# Example usage:
# analyze_multilabel_class_distribution(combined_labels, class_names=mlb.classes_)

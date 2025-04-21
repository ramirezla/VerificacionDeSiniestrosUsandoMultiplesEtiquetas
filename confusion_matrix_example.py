import numpy as np

def multilabel_to_binary_indicator(label_lists, num_classes):
    """
    Convert list of label indices per sample to binary indicator matrix.
    label_lists: list of lists, each inner list contains label indices for a sample
    num_classes: total number of classes
    Returns: numpy array of shape (num_samples, num_classes) with 0/1 indicators
    """
    binary_matrix = np.zeros((len(label_lists), num_classes), dtype=int)
    for i, labels in enumerate(label_lists):
        for label in labels:
            if 0 <= label-1 < num_classes:  # assuming labels start at 1
                binary_matrix[i, label-1] = 1
    return binary_matrix

# Example usage with your data:

# Suppose you have loaded your test dataset as test_df with column 'dannos' containing lists of labels

# Number of damage classes
num_danos = len(label_to_cls_danos)

# Convert true labels to binary indicator
y_true = multilabel_to_binary_indicator(test_df['dannos'], num_danos)

# Suppose y_pred is your model predictions in the same format (list of label indices per sample)
# For demonstration, let's create dummy predictions (replace with your model output)
y_pred = multilabel_to_binary_indicator(test_df['dannos'], num_danos)  # dummy: perfect prediction

# Class names
class_names = list(label_to_cls_danos.values())

# Now call the plotting function from plotting_code_snippets.py
# from plotting_code_snippets import plot_confusion_matrix_multilabel
# plot_confusion_matrix_multilabel(y_true.tolist(), y_pred.tolist(), class_names)

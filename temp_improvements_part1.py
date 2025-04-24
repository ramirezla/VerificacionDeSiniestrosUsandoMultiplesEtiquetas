# Advanced Oversampling and Adaptive Focal Loss Parameter Tuning for Class Imbalance Handling

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

# Example: Applying SMOTE or ADASYN to multi-label encoded dataset
import numpy as np

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

# Example usage:
# Assuming you have feature matrix X and multi-label binary matrix y
# X_resampled, y_resampled = apply_oversampling(X, y, method='SMOTE')

# Adaptive Focal Loss with per-class alpha and gamma tuning
class AdaptiveFocalLossPerClass(nn.Module):
    def __init__(self, alpha=None, gamma=None, reduction='mean'):
        """
        alpha: tensor of shape (num_classes,) or scalar
        gamma: tensor of shape (num_classes,) or scalar
        """
        super().__init__()
        if alpha is None:
            self.alpha = 0.75
        else:
            self.alpha = alpha
        if gamma is None:
            self.gamma = 2.0
        else:
            self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits tensor of shape (batch_size, num_classes)
        targets: binary tensor of shape (batch_size, num_classes)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        if isinstance(self.alpha, torch.Tensor):
            alpha_factor = self.alpha.unsqueeze(0) * targets + (1 - self.alpha.unsqueeze(0)) * (1 - targets)
        else:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        if isinstance(self.gamma, torch.Tensor):
            focal_weight = (1 - pt) ** self.gamma.unsqueeze(0)
        else:
            focal_weight = (1 - pt) ** self.gamma

        loss = alpha_factor * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Example of setting per-class alpha and gamma based on class frequency
def compute_alpha_gamma(class_counts, total_samples, base_alpha=0.75, base_gamma=2.0):
    """
    Compute per-class alpha and gamma values inversely proportional to class frequency.
    class_counts: list or array of counts per class
    total_samples: total number of samples
    Returns: alpha tensor, gamma tensor
    """
    freq = np.array(class_counts) / total_samples
    alpha = 1.0 - freq  # less frequent classes get higher alpha
    alpha = base_alpha * alpha / alpha.max()  # normalize to base_alpha max
    gamma = base_gamma * (1.0 - freq)  # higher gamma for rare classes
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32)
    return alpha_tensor, gamma_tensor

# Integration example inside your training notebook (DetectarDannosPartesSugerenciasUsandoMultiplesEtiquetas_V7_01.ipynb):

"""
# After loading your training dataset and before training:

# 1. Compute class counts for parts, damages, suggestions
part_counts = [number_of_samples_per_part_class]  # Replace with actual counts
damage_counts = [number_of_samples_per_damage_class]  # Replace with actual counts
suggestion_counts = [number_of_samples_per_suggestion_class]  # Replace with actual counts

total_samples = len(multi_train)  # or your training dataset length

# 2. Compute alpha and gamma for each task
alpha_parts, gamma_parts = compute_alpha_gamma(part_counts, total_samples)
alpha_damages, gamma_damages = compute_alpha_gamma(damage_counts, total_samples)
alpha_suggestions, gamma_suggestions = compute_alpha_gamma(suggestion_counts, total_samples)

# 3. Define loss functions with per-class parameters
criterion_parts = AdaptiveFocalLossPerClass(alpha=alpha_parts, gamma=gamma_parts)
criterion_damages = AdaptiveFocalLossPerClass(alpha=alpha_damages, gamma=gamma_damages)
criterion_suggestions = nn.CrossEntropyLoss()  # or adapt similarly if needed

# 4. Modify your balanced_multi_task_loss function to use these criteria:

def balanced_multi_task_loss(outputs, targets):
    parts_loss = criterion_parts(outputs['parts'], targets['parts'].float())
    damages_loss = criterion_damages(outputs['damages'], targets['damages'].float())
    suggestions_loss = criterion_suggestions(outputs['suggestions'], targets['suggestions'].argmax(dim=1))
    return 0.5 * parts_loss + 0.3 * damages_loss + 0.2 * suggestions_loss

# 5. Use this loss function in your training loop.
"""

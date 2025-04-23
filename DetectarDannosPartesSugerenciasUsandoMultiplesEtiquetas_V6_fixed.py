import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate_multi_label_fixed(model, data_loader, thresholds=None):
    if thresholds is None:
        thresholds = {
            'parts': 0.5,
            'damages': 0.5,
            'suggestions': 0.5
        }

    model.eval()
    parts_preds = []
    parts_targets = []
    damages_preds = []
    damages_targets = []
    suggestions_preds = []
    suggestions_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(model.base_model.conv1.weight.device)
            outputs = model(inputs)

            parts_preds.append((torch.sigmoid(outputs['parts']) > thresholds['parts']).float().cpu())
            parts_targets.append(targets['parts'].cpu())

            damages_preds.append((torch.sigmoid(outputs['damages']) > thresholds['damages']).float().cpu())
            damages_targets.append(targets['damages'].cpu())

            suggestions_preds.append(torch.softmax(outputs['suggestions'], dim=1).cpu())
            suggestions_targets.append(targets['suggestions'].cpu())

    parts_preds = torch.cat(parts_preds)
    parts_targets = torch.cat(parts_targets)
    damages_preds = torch.cat(damages_preds)
    damages_targets = torch.cat(damages_targets)
    suggestions_preds = torch.cat(suggestions_preds)
    suggestions_targets = torch.cat(suggestions_targets)

    def calculate_metrics(preds, targets, task_type='multilabel'):
        # preds are already thresholded for multilabel tasks
        if task_type == 'multilabel':
            accuracy = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average='macro', zero_division=0)
        else:
            preds = preds.argmax(dim=1)
            targets = targets.argmax(dim=1)
            accuracy = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average='macro', zero_division=0)
        return {'accuracy': accuracy, 'f1_macro': f1}

    metrics = {
        'parts': calculate_metrics(parts_preds, parts_targets, 'multilabel'),
        'damages': calculate_metrics(damages_preds, damages_targets, 'multilabel'),
        'suggestions': calculate_metrics(suggestions_preds, suggestions_targets, 'multiclass')
    }

    return metrics

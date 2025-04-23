# =============================================
# IMPORTS
# =============================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from collections import Counter
from tqdm import tqdm
import ast
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
# =============================================
# CONFIGURACIÓN MEJORADA
# =============================================
# Configuración adicional
CLASS_WEIGHTS = True
FOCAL_LOSS = True
AUGMENTATION = True
EARLY_STOPPING = True
USE_TENSORBOARD = True
# Hiperparámetros optimizados
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 100  # Aumentado para permitir más aprendizaje
MIN_SAMPLES_PER_CLASS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20  # Aumentada para early stopping
# Variables para guarda las métricas para graficar
train_loss_history = []
val_metric_history = []
# =============================================
# DICCIONARIOS COMPLETOS DE MAPEO
# =============================================
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
# Diccionario para Tipos de Daño (completo)
label_to_cls_danos = {
    1: "Abolladura",
    2: "Deformación",
    3: "Desprendimiento",
    4: "Fractura",
    5: "Rayón",
    6: "Rotura"
}
# Diccionario para Sugerencia (completo)
label_to_cls_sugerencia = {
    1: "Reparar",
    2: "Reemplazar"
}
# =============================================
# Cargar los datasets
# =============================================
multi_train = pd.read_csv('data/fotos_siniestros/datasets/multi_train.csv', sep='|')
multi_val = pd.read_csv('data/fotos_siniestros/datasets/multi_val.csv', sep='|')
multi_test = pd.read_csv('data/fotos_siniestros/datasets/multi_test.csv', sep='|')
# =============================================
# DATASET
# =============================================
class BalancedMultiLabelDamageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
        # Convertir strings a listas
        for col in ['parts', 'damages', 'suggestions']:
            self.data[col] = self.data[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # Mapeo de clases
        self.part_to_idx = {part: idx for idx, part in label_to_cls_piezas.items()}
        self.damage_to_idx = {damage: idx for idx, damage in label_to_cls_danos.items()}
        self.suggestion_to_idx = {sug: idx for idx, sug in label_to_cls_sugerencia.items()}
        # Binarizadores mejorados
        self.part_binarizer = MultiLabelBinarizer(classes=sorted(self.part_to_idx.values()))
        self.damage_binarizer = MultiLabelBinarizer(classes=sorted(self.damage_to_idx.values()))
        self.suggestion_binarizer = MultiLabelBinarizer(classes=sorted(self.suggestion_to_idx.values()))
        # Calcular pesos mejorados
        self.part_weights = self._calculate_weights('part')
        self.damage_weights = self._calculate_weights('damage')
        self.suggestion_weights = self._calculate_weights('suggestion')
    def _calculate_weights(self, task):
        """Versión mejorada con square root inverse frequency"""
        all_labels = [label for labels in self.data[f'{task}s']
                     for label in labels if label in getattr(self, f'{task}_to_idx')]
        if not all_labels:
            return torch.ones(len(getattr(self, f'{task}_to_idx')), dtype=torch.float32).to(DEVICE)
        counts = Counter(all_labels)
        total = len(all_labels)
        weights = {cls: 1/np.sqrt(count/total) for cls, count in counts.items()}
        # Peso mínimo de 1.0 para clases no presentes
        return torch.tensor([weights.get(cls, 1.0) for cls in sorted(getattr(self, f'{task}_to_idx').values())],
                          dtype=torch.float32).to(DEVICE)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = ""  # Inicializamos la variable
        try:
            # Verificamos que el índice sea válido
            if idx >= len(self.data):
                raise IndexError(f"Índice {idx} fuera de rango (tamaño del dataset: {len(self.data)})")

            img_path = os.path.join(self.img_dir, self.data.iloc[idx]['Imagen'])
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error al cargar imagen (índice {idx}): {str(e)}")
            if not img_path:  # Si no se pudo obtener la ruta
                img_path = f"Índice inválido: {idx}"
            print(f"Ruta problemática: {img_path}")
            image = torch.zeros(3, 224, 224)  # Imagen dummy
        # Resto del código para procesar etiquetas...
        parts = torch.zeros(len(self.part_to_idx))
        for part in self.data.iloc[idx]['parts']:
            if part in self.part_to_idx:
                parts[self.part_to_idx[part]] = 1
        damages = torch.zeros(len(self.damage_to_idx))
        for damage in self.data.iloc[idx]['damages']:
            if damage in self.damage_to_idx:
                damages[self.damage_to_idx[damage]] = 1
        suggestions = torch.zeros(len(self.suggestion_to_idx))
        for sug in self.data.iloc[idx]['suggestions']:
            if sug in self.suggestion_to_idx:
                suggestions[self.suggestion_to_idx[sug]] = 1
        if self.transform:
            image = self.transform(image)
        return image, {
            'parts': parts,
            'damages': damages,
            'suggestions': suggestions
        }
class MultiLabelDamageClassifier(nn.Module):
    def __init__(self, num_parts, num_damages, num_suggestions):
        super().__init__()
        # Base model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Forma actualizada
        # Congelar capas base inicialmente
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Capa compartida
        num_features = self.base_model.fc.in_features
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Cabezales de clasificación
        self.parts_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_parts)
        )
        self.damages_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_damages)
        )
        self.suggestions_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_suggestions)
        )
        # Inicialización
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.base_model.conv1(x)
        features = self.base_model.bn1(features)
        features = self.base_model.relu(features)
        features = self.base_model.maxpool(features)
        features = self.base_model.layer1(features)
        features = self.base_model.layer2(features)
        features = self.base_model.layer3(features)
        features = self.base_model.layer4(features)
        features = self.base_model.avgpool(features)
        features = torch.flatten(features, 1)
        shared_features = self.shared_fc(features)
        return {
            'parts': self.parts_head(shared_features),
            'damages': self.damages_head(shared_features),
            'suggestions': self.suggestions_head(shared_features)
        }
    def unfreeze_layers(self, num_layers=3):
        """Descongela capas para fine-tuning"""
        # Capas base
        for name, param in self.base_model.named_parameters():
            if any(f'layer{i}.' in name for i in range(4, 4-num_layers, -1)):
                param.requires_grad = True
# =============================================
# DATA AUGMENTATION
# =============================================
def get_transforms():
    # Transformaciones base
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if AUGMENTATION:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, shear=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = base_transform
    return {
        'train': train_transform,
        'val': base_transform,
        'test': base_transform
    }
# =============================================
# Recolecta las imagenes
# =============================================
def collate_fn(batch):
    # Filtrar None (imágenes que fallaron al cargar)
    batch = [b for b in batch if b is not None]

    # Si todo el batch falló, retornar un batch dummy
    if len(batch) == 0:
        dummy_image = torch.zeros(3, 224, 224)
        dummy_target = {
            'parts': torch.zeros(len(label_to_cls_piezas)),
            'damages': torch.zeros(len(label_to_cls_danos)),
            'suggestions': torch.zeros(len(label_to_cls_sugerencia))
        }
        return dummy_image.unsqueeze(0), dummy_target
    # Procesamiento normal
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    batch_targets = {
        'parts': torch.stack([t['parts'] for t in targets], dim=0),
        'damages': torch.stack([t['damages'] for t in targets], dim=0),
        'suggestions': torch.stack([t['suggestions'] for t in targets], dim=0)
    }
    return images, batch_targets
# =============================================
# FUNCIONES DE PÉRDIDA
# =============================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets, weights=None):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if weights is not None:
            # Aplicar pesos por clase
            class_weights = weights.expand_as(targets)
            F_loss = F_loss * class_weights
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss
def balanced_multi_label_loss(outputs, targets, weights):
    # Configuración mejorada de pérdidas
    parts_loss_fn = FocalLoss(alpha=0.25, gamma=2.0) if FOCAL_LOSS else nn.BCEWithLogitsLoss()
    damages_loss_fn = FocalLoss(alpha=0.5, gamma=2.0) if FOCAL_LOSS else nn.BCEWithLogitsLoss()
    # Pérdidas con pesos
    parts_loss = parts_loss_fn(
        outputs['parts'],
        targets['parts'].float(),
        weights=weights['parts'] if CLASS_WEIGHTS else None
    )
    damages_loss = damages_loss_fn(
        outputs['damages'],
        targets['damages'].float(),
        weights=weights['damages'] if CLASS_WEIGHTS else None
    )
    # Sugerencias con label smoothing
    suggestions_loss = F.cross_entropy(
        outputs['suggestions'],
        targets['suggestions'].argmax(dim=1),
        weight=weights['suggestions'] if CLASS_WEIGHTS else None,
        label_smoothing=0.1
    )
    return 0.4 * parts_loss + 0.4 * damages_loss + 0.2 * suggestions_loss
# Nueva funcion
def evaluate_multi_label(model, data_loader, thresholds=None):
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
# =============================================
# CLASE EARLY STOPPING
# =============================================
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, warmup=20):
        self.patience = patience
        self.delta = delta
        self.warmup = warmup
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.epoch = 0
    def __call__(self, current_metric):
        self.epoch += 1
        if self.epoch < self.warmup:  # Periodo de calentamiento
            return False
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric < self.best_metric + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = current_metric
            self.counter = 0
        return self.early_stop
# =============================================
# ESTRATEGIAS DE ENTRENAMIENTO MEJORADAS
# =============================================
def train_model_improved():
    # Configuración inicial
    writer = SummaryWriter() if USE_TENSORBOARD else None
    data_transforms = get_transforms()
    # Crear datasets
    train_dataset = BalancedMultiLabelDamageDataset(multi_train, '../data/fotos_siniestros/', data_transforms['train'])
    val_dataset = BalancedMultiLabelDamageDataset(multi_val, '../data/fotos_siniestros/', data_transforms['val'])
    # Sampler mejorado
    def get_sampler(dataset):
        """Sampler que considera múltiples tareas"""
        labels = []
        for idx in range(len(dataset)):
            # Combinar etiquetas de partes y daños para balanceo
            combined = tuple(dataset[idx][1]['parts'].nonzero().flatten().tolist() +
                           dataset[idx][1]['damages'].nonzero().flatten().tolist())
            labels.append(combined)
        class_counts = Counter(labels)
        class_weights = {cls: 1./count for cls, count in class_counts.items()}
        sample_weights = [class_weights[cls] for cls in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=get_sampler(train_dataset),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    # Inicializar modelo
    model = MultiLabelDamageClassifier(
        num_parts=len(label_to_cls_piezas),
        num_damages=len(label_to_cls_danos),
        num_suggestions=len(label_to_cls_sugerencia)
    ).to(DEVICE)
    # Early Stopping
    early_stopper = EarlyStopping(patience=PATIENCE, delta=0.001, warmup=20) if EARLY_STOPPING else None
    # =============================================
    # FASE 1: Entrenamiento solo de cabezales
    # =============================================
    print("\n" + "="*60)
    print("FASE 1: Entrenamiento solo de cabezales (10 épocas)")
    print("="*60)
    # Congelar todo excepto los cabezales
    for param in model.parameters():
        param.requires_grad = False
    for param in model.parts_head.parameters():
        param.requires_grad = True
    for param in model.damages_head.parameters():
        param.requires_grad = True
    for param in model.suggestions_head.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW([
        {'params': model.parts_head.parameters(), 'lr': 1e-3},
        {'params': model.damages_head.parameters(), 'lr': 1e-3},
        {'params': model.suggestions_head.parameters(), 'lr': 1e-3}
    ], weight_decay=WEIGHT_DECAY)
    for epoch in range(10):
        # Entrenamiento
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = balanced_multi_label_loss(outputs, targets, {
                'parts': train_dataset.part_weights,
                'damages': train_dataset.damage_weights,
                'suggestions': train_dataset.suggestion_weights
            })
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validación usando thresholds personalizados
        ## val_metrics = evaluate_multi_label(model, val_loader, thresholds={'parts': 0.3,'damages': 0.2,'suggestions': 0.5})
        val_metrics = evaluate_multi_label(model, val_loader)
        current_metric = 0.4*val_metrics['parts']['f1_macro'] + 0.4*val_metrics['damages']['f1_macro'] + 0.2*val_metrics['suggestions']['f1_macro']
        # Logging
        print(f"\nEpoch {epoch+1}/10")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation Metric: {current_metric:.4f}")
        print("Detailed Metrics:")
        for task, metrics in val_metrics.items():
            print(f"  {task:12} - Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
        if writer:
            writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
            writer.add_scalar('Metric/val', current_metric, epoch)
            for task, metrics in val_metrics.items():
                writer.add_scalar(f'Accuracy/{task}', metrics['accuracy'], epoch)
                writer.add_scalar(f'F1/{task}', metrics['f1_macro'], epoch)
    # =============================================
    # FASE 2: Fine-tuning parcial
    # =============================================
    print("\n" + "="*60)
    print("FASE 2: Fine-tuning parcial (20 épocas)")
    print("="*60)
    # Descongelar capas superiores y shared_fc
    model.unfreeze_layers(3)  # Descongela las 3 últimas capas de ResNet
    for param in model.shared_fc.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW([
        {'params': model.shared_fc.parameters(), 'lr': 1e-4},
        {'params': model.parts_head.parameters(), 'lr': 1e-4},
        {'params': model.damages_head.parameters(), 'lr': 1e-4},
        {'params': model.suggestions_head.parameters(), 'lr': 1e-4},
        {'params': model.base_model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.base_model.layer3.parameters(), 'lr': 1e-5}
    ], weight_decay=WEIGHT_DECAY)
    for epoch in range(10, 30):
        # Entrenamiento
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = balanced_multi_label_loss(outputs, targets, {
                'parts': train_dataset.part_weights,
                'damages': train_dataset.damage_weights,
                'suggestions': train_dataset.suggestion_weights
            })
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validación usando thresholds personalizados
        ## val_metrics = evaluate_multi_label(model, val_loader, thresholds={'parts': 0.3,'damages': 0.2,'suggestions': 0.5})
        val_metrics = evaluate_multi_label(model, val_loader)
        current_metric = 0.4*val_metrics['parts']['f1_macro'] + 0.4*val_metrics['damages']['f1_macro'] + 0.2*val_metrics['suggestions']['f1_macro']
        # Early Stopping
        if EARLY_STOPPING:
            early_stopper(current_metric)
            if early_stopper.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1} with metric {current_metric:.4f}")
                break
        # Logging
        print(f"\nEpoch {epoch+1}/30")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation Metric: {current_metric:.4f}")
        print("Detailed Metrics:")
        for task, metrics in val_metrics.items():
            print(f"  {task:12} - Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
        if writer:
            writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
            writer.add_scalar('Metric/val', current_metric, epoch)
            for task, metrics in val_metrics.items():
                writer.add_scalar(f'Accuracy/{task}', metrics['accuracy'], epoch)
                writer.add_scalar(f'F1/{task}', metrics['f1_macro'], epoch)
    # =============================================
    # FASE 3: Fine-tuning completo (opcional)
    # =============================================
    if not EARLY_STOPPING or not early_stopper.early_stop:
        print("\n" + "="*60)
        print("FASE 3: Fine-tuning completo")
        print("="*60)
        # Descongelar todo el modelo
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WEIGHT_DECAY)
        for epoch in range(30, NUM_EPOCHS):
            # Entrenamiento
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in targets.items()}
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = balanced_multi_label_loss(outputs, targets, {
                    'parts': train_dataset.part_weights,
                    'damages': train_dataset.damage_weights,
                    'suggestions': train_dataset.suggestion_weights
                })
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # Validación usando thresholds personalizados
            ## val_metrics = evaluate_multi_label(model, val_loader, thresholds={'parts': 0.3,'damages': 0.2,'suggestions': 0.5})
            val_metrics = evaluate_multi_label(model, val_loader)
            current_metric = 0.4*val_metrics['parts']['f1_macro'] + 0.4*val_metrics['damages']['f1_macro'] + 0.2*val_metrics['suggestions']['f1_macro']
            # Early Stopping
            if EARLY_STOPPING:
                early_stopper(current_metric)
                if early_stopper.early_stop:
                    print(f"\nEarly stopping at epoch {epoch+1} with metric {current_metric:.4f}")
                    break
            # Logging
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Validation Metric: {current_metric:.4f}")
            print("Detailed Metrics:")
            for task, metrics in val_metrics.items():
                print(f"  {task:12} - Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
            if writer:
                writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
                writer.add_scalar('Metric/val', current_metric, epoch)
                for task, metrics in val_metrics.items():
                    writer.add_scalar(f'Accuracy/{task}', metrics['accuracy'], epoch)
                    writer.add_scalar(f'F1/{task}', metrics['f1_macro'], epoch)
    if writer:
        writer.close()
    return model
# =============================================
# FUNCIONES AUXILIARES PARA EL ENTRENAMIENTO
# =============================================
def train_epoch(model, loader, optimizer, epoch, writer=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}', leave=False)
    for inputs, targets in progress_bar:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = balanced_multi_label_loss(outputs, targets, {
            'parts': loader.dataset.part_weights,
            'damages': loader.dataset.damage_weights,
            'suggestions': loader.dataset.suggestion_weights
        })
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        if writer:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(loader) + len(progress_bar))
    return total_loss / len(loader)
def log_metrics(epoch, metrics, writer):
    combined_metric = 0.4*metrics['parts']['f1_macro'] + 0.4*metrics['damages']['f1_macro'] + 0.2*metrics['suggestions']['f1_macro']
    print(f"\nEpoch {epoch+1} Metrics:")
    print(f"Combined Metric: {combined_metric:.4f}")
    for task, values in metrics.items():
        print(f"{task.capitalize():12} - Acc: {values['accuracy']:.4f} | F1: {values['f1_macro']:.4f}")
    if writer:
        writer.add_scalar('Metric/combined', combined_metric, epoch)
        for task, values in metrics.items():
            writer.add_scalar(f'Accuracy/{task}', values['accuracy'], epoch)
            writer.add_scalar(f'F1/{task}', values['f1_macro'], epoch)
# =============================================
# EJECUCIÓN PRINCIPAL
# =============================================
# Last Execution 5:09:58 PM
# Execution Time 139m 36.1s
# Overhead Time 71m 10.0s
# Render Times
# VS Code Builtin Notebook Output Renderer 2ms
if __name__ == '__main__':
    # Inicializar early stopper
    early_stopper = EarlyStopping(patience=PATIENCE, delta=0.001, warmup=20) if EARLY_STOPPING else None
    # Entrenar modelo
    trained_model = train_model_improved()
    # Guardar modelo
    torch.save(trained_model.state_dict(), 'DetectarDannosPartesSugerenciasUsandoMultiplesEtiquetas_V601.pth')

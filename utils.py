import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from model_factory import TASK_CONFIGURATIONS  # Needed for task name mapping

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def multi_task_collate_fn(batch):
    """
    Custom collate function to handle different label shapes in multi-task learning.
    Images are stacked; labels and task_ids remain as lists.
    """
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    task_ids = [item['task_id'] for item in batch]
    
    # Stack images as they have consistent dimensions
    images = torch.stack(images, 0)
    
    return {'image': images, 'label': labels, 'task_id': task_ids}

# class DetectionLoss(nn.Module):
#     """A simplified loss function for object detection (Cls + Reg)."""
#     def __init__(self, classification_weight=1.0, box_regression_weight=8.0):
#         super().__init__()
#         self.classification_loss = nn.BCEWithLogitsLoss()
#         self.box_regression_loss = nn.L1Loss()
#         self.cls_w, self.box_w = classification_weight, box_regression_weight

#     def forward(self, predictions, targets):
#         # predictions: [B, 5], targets: [B, 4]
#         pred_boxes, pred_scores = predictions[:, :4], predictions[:, 4].squeeze(-1)
        
#         # Filter valid targets (dummy targets are usually -1)
#         valid_indices = (targets[:, 0] >= 0).view(-1)
#         if not valid_indices.any(): 
#             return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
#         cls_loss = self.classification_loss(pred_scores[valid_indices], torch.ones_like(pred_scores)[valid_indices])
#         box_loss = self.box_regression_loss(pred_boxes[valid_indices], targets[valid_indices])
        
#         return self.cls_w * cls_loss + self.box_w * box_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target_boxes):
        # pred: [B, 5, H, W] → [x1, y1, x2, y2, obj_logit]
        # target_boxes: [B, 4] → normalized (x1, y1, x2, y2) in [0,1]
        B, _, H, W = pred.shape
        device = pred.device

        # 创建目标：全0，只在GT位置设为1（objectness）和真实框（bbox）
        tgt_obj = torch.zeros(B, 1, H, W, device=device)
        tgt_bbox = torch.zeros(B, 4, H, W, device=device)

        for b in range(B):
            x1, y1, x2, y2 = target_boxes[b]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            gi = min(int(cx * W), W - 1)  # x → width
            gj = min(int(cy * H), H - 1)  # y → height
            tgt_obj[b, 0, gj, gi] = 1.0
            tgt_bbox[b, :, gj, gi] = target_boxes[b]

        pred_bbox = pred[:, :4]
        pred_obj = pred[:, 4:5]

        # 损失1：objectness（用 BCE）
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, tgt_obj, reduction='mean')
        # 损失2：bbox（只算正样本）
        bbox_loss = (self.bbox_loss(pred_bbox, tgt_bbox) * tgt_obj).mean()

        return obj_loss + bbox_loss

# --- Metric Calculations ---

def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return accuracy_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_dice_coefficient(y_true, y_pred_logits):
    y_pred_mask = torch.argmax(y_pred_logits, dim=1)
    num_classes = y_pred_logits.shape[1]
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)
    y_pred_one_hot = F.one_hot(y_pred_mask, num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(y_true_one_hot[:, 1:] * y_pred_one_hot[:, 1:])
    union = torch.sum(y_true_one_hot[:, 1:]) + torch.sum(y_pred_one_hot[:, 1:])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

def calculate_mae(y_true, y_pred, image_size=(256, 256)):
    h, w = image_size
    y_true_px = y_true.cpu().numpy().copy()
    y_pred_px = y_pred.cpu().numpy().copy()
    y_true_px[:, 0::2] *= w; y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w; y_pred_px[:, 1::2] *= h
    return np.mean(np.abs(y_true_px - y_pred_px))

def calculate_iou(y_true, y_pred):
    y_true = y_true.cpu().numpy(); y_pred = y_pred.cpu().numpy()
    batch_ious = []
    for i in range(y_true.shape[0]):
        box_true, box_pred = y_true[i], y_pred[i]
        xA = max(box_true[0], box_pred[0]); yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2]); yB = min(box_true[3], box_pred[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_true_area = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
        box_pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        union_area = box_true_area + box_pred_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        batch_ious.append(iou)
    return np.mean(batch_ious)

def evaluate(model, val_loader, device):
    """
    Evaluation loop supporting multi-task batches.
    """
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]")
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label']
            task_ids = batch['task_id']

            unique_tasks_in_batch = set(task_ids)

            for task_id in unique_tasks_in_batch:
                task_indices = [i for i, t_id in enumerate(task_ids) if t_id == task_id]
                task_images = images[task_indices]
                
                # Extract and stack labels for the current task
                task_labels_list = [labels[i] for i in task_indices]
                task_labels = torch.stack(task_labels_list, 0)
                
                outputs = model(task_images, task_id=task_id)
                task_name = task_id_to_name[task_id]
                
                if task_name == 'classification':
                    task_metrics[task_id]['Accuracy'].append(calculate_accuracy(task_labels, outputs))
                    task_metrics[task_id]['F1-Score'].append(calculate_f1_score(task_labels, outputs))
                
                elif task_name == 'segmentation':
                    task_metrics[task_id]['Dice'].append(calculate_dice_coefficient(task_labels.to(device), outputs))
                
                elif task_name == 'Regression':
                    task_metrics[task_id]['MAE (pixels)'].append(calculate_mae(task_labels, outputs))
                
                elif task_name == 'detection':
                    # Logic to extract best bounding box from grid prediction
                    # batch_size, _, h, w = outputs.shape
                    # scores = outputs[:, 4, :, :].view(batch_size, -1) 
                    # _, best_indices = torch.max(scores, dim=1)
                    
                    # best_h = best_indices // w
                    # best_w = best_indices % w
                    
                    # final_boxes = torch.zeros((batch_size, 4), device=device)
                    # for i in range(batch_size):
                    #     final_boxes[i] = outputs[i, :4, best_h[i], best_w[i]]
                    B, C, H, W = outputs.shape
                    assert C == 4, f"Expected 4 channels for detection, got {C}"
                    
                    # 聚合方式必须与训练一致！这里用 mean（推荐）或 max
                    pred_normalized = outputs.view(B, 4, -1).mean(dim=2)  # (B, 4) in [0,1]
                
                    # Convert [cx, cy, w, h] -> [x1, y1, x2, y2]
                    def cxcywh_to_xyxy(boxes):
                        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        return torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)
                
                    pred_boxes_xyxy = cxcywh_to_xyxy(pred_normalized)
                    true_boxes_xyxy = cxcywh_to_xyxy(task_labels.to(device))
                    
                    task_metrics[task_id]['IoU'].append(calculate_iou(true_boxes_xyxy, pred_boxes_xyxy))

    results = []
    sorted_task_ids = sorted(list(task_id_to_name.keys()))
    for task_id in sorted_task_ids:
        if task_id in task_metrics:
            task_name = task_id_to_name[task_id]
            result_row = {'Task ID': task_id, 'Task Name': task_name}
            for metric_name, values in task_metrics[task_id].items():
                result_row[metric_name] = np.mean(values)
            results.append(result_row)
    return pd.DataFrame(results)
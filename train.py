import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random
import cv2
# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn, 
    evaluate, 
    DetectionLoss, 
    set_seed
)
from torchvision.utils import draw_bounding_boxes
# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 30 
DATA_ROOT_PATH = './train'
ENCODER = 'convnext_base.fb_in22k_ft_in1k_384'
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.05

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Data loading and splitting
    # Training transforms with augmentation
    train_seg_transforms = A.Compose([
        A.Resize(384, 384), 

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(
        shift_limit=0.1,      # 平移 ±10%
        scale_limit=0.2,      # 缩放 0.8～1.2x
        rotate_limit=15,      # 旋转 ±15°
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT,  # 避免插值 artifacts
        value=0, mask_value=0
        ),
    
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
    
        A.CoarseDropout(
           max_holes=8,
           max_height=32,
           max_width=32,
           min_holes=1,
           min_height=8,
           min_width=8,
           fill_value=0,
           p=0.2
        ), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    train_class_transforms = A.Compose([
        A.Resize(384, 384), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    
    # Validation transforms without augmentation
    val_transforms = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Create full dataset to get indices
    temp_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms_seg=train_seg_transforms,transforms_class=train_class_transforms)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size
    
    # Split indices
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = list(range(dataset_size))
    train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)
    
    # Create separate datasets with different transforms
    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms_seg=train_seg_transforms,transforms_class=train_class_transforms)
    val_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms_seg=val_transforms,transforms_class=val_transforms)
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Fix dataframe reference for subset
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices.indices].reset_index(drop=True)
    
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=8,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Model and loss setup
    model = MultiTaskModelFactory(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, task_configs=TASK_CONFIGURATIONS).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params / 1e6:.1f}MB")
    
    loss_functions = {
        'segmentation': {
            'dice': smp_losses.DiceLoss(mode='multiclass'),
            'focal': smp_losses.FocalLoss(mode='multiclass')
        },
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(), 
        'detection': nn.SmoothL1Loss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    TASK_LOSS_WEIGHTS = {
        # Regression
        'FUGC': 1.0,
        'IUGC': 1.0,
        'fetal_femur': 2.0,
        'spinal_cord_injury_loc': 1.0,
    
        # Classification
        'breast_2cls': 1.0,
        'breast_3cls': 1.0,
        'lung_2cls': 1.0,            
        'lung_disease_3cls': 1.0,
        'organ_cls': 1.0,
        'liver_lesion_2cls': 1.0,
        'fetal_plane_cls': 1.0,
        'fetal_head_pos_cls': 1.0,
        'fetal_sacral_pos_cls': 1.0,
    
        # Segmentation
        'breast_lesion': 1.0,
        'cardiac_multi': 1.0,
        'carotid_artery': 1.0,
        'cervix': 1.0,
        'cervix_multi': 1.0,
        'fetal_abdomen_multi': 1.0,
        'fetal_head': 1.0,
        'fetal_heart': 1.0,
        'head_symphysis_multi': 1.0,
        'lung': 1.0,
        'ovary_tumor': 1.0,
        'thyroid_nodule': 2.0,
    
        # Detection
        'thyroid_nodule_det': 2.0,
        'uterine_fibroid_det': 1.0,
    }

    # Optimization setup
    print("\n--- Setting parameter groups ---")
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 1},
        {'params': model.backbone_1.parameters(), 'lr': LEARNING_RATE * 1},
    ]
    print(f"  - Shared Encoder                 -> LR: {LEARNING_RATE * 1}")

    
    for task_id, head in model.heads.items():
        lr_multiplier = 10.0
        current_lr = LEARNING_RATE * lr_multiplier
        param_groups.append({'params': head.parameters(), 'lr': current_lr})
        print(f"  - Task Head '{task_id:<25}' -> LR: {current_lr}")

    optimizer = optim.AdamW(param_groups)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    print("\n--- Cosine Annealing Scheduler configured ---")

    best_val_score = -float('inf')
    print("\n" + "="*50 + "\n--- Start Training ---")
    dice_loss_weight = 0.5
    focal_loss_weight = 0.5
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # Manually stack labels list to tensor
            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]
            outputs = model(images, task_id=current_task_id)
            
            # Grid-based detection logic
            if task_name == 'detection':
                B, C, H, W = outputs.shape
                final_outputs = outputs.view(B, C, -1).mean(dim=2)
            else:
                final_outputs = outputs
            
            if task_name == 'segmentation':
                dice_output = loss_functions[task_name]['dice'](final_outputs, labels)
                focal_output = loss_functions[task_name]['focal'](final_outputs, labels)
                loss = dice_loss_weight * dice_output + focal_loss_weight * focal_output
            else:
                loss = loss_functions[task_name](final_outputs, labels)

            loss = TASK_LOSS_WEIGHTS[current_task_id] * loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses[current_task_id].append(loss.item())
            loop.set_postfix(loss=loss.item(), task=current_task_id, lr=scheduler.get_last_lr()[0])

        # Train reporting
        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        print("-" * 40)

        # Validation
        val_results_df = evaluate(model, val_loader, device)
        
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! Score improved to: {best_val_score:.4f}\n")
        
        # Update scheduler
        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
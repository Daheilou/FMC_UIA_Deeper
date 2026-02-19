import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List, Dict
import timm

# Task configuration list
TASK_CONFIGURATIONS = [
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'FUGC'},
    {'task_name': 'Regression', 'num_classes': 3, 'task_id': 'IUGC'},
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'fetal_femur'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'breast_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'breast_3cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_head_pos_cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'fetal_plane_cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_sacral_pos_cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'liver_lesion_2cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'lung_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'lung_disease_3cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'organ_cls'},
    {'task_name': 'detection', 'num_classes': 4, 'task_id': 'spinal_cord_injury_loc'},
    {'task_name': 'detection', 'num_classes': 4, 'task_id': 'thyroid_nodule_det'},
    {'task_name': 'detection', 'num_classes': 4, 'task_id': 'uterine_fibroid_det'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'breast_lesion'},
    {'task_name': 'segmentation', 'num_classes': 4, 'task_id': 'cardiac_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'carotid_artery'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'cervix'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'cervix_multi'},
    {'task_name': 'segmentation', 'num_classes': 5, 'task_id': 'fetal_abdomen_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_head'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_heart'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'head_symphysis_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'lung'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'ovary_tumor'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'thyroid_nodule'},
]

# Task specific heads

# ======================
# Custom FPN-like Decoder (lightweight, compatible with timm features)
# ======================
class TimmFPNDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=256):
        super().__init__()
        self.encoder_channels = encoder_channels  # e.g., [128, 128, 256, 512, 1024] for convnext_base
        self.num_stages = len(encoder_channels)
        
        # Lateral convolutions (1x1 to unify channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, decoder_channels, kernel_size=1)
            for in_ch in encoder_channels
        ])
        
        # Top-down path: upsampling + addition
        self.upsample_convs = nn.ModuleList([
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1)
            for _ in range(self.num_stages - 1)
        ])
        
        self.out_channels = decoder_channels

    def forward(self, features: List[torch.Tensor]):
        # features: [C1, C2, C3, C4, C5] from timm (low to high level)
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = nn.functional.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], 
                mode='nearest'
            )
            laterals[i-1] = laterals[i-1] + upsampled
            laterals[i-1] = self.upsample_convs[i-1](laterals[i-1])
        
        # Return the highest resolution fused feature (P1)
        return laterals[0]  # shape: [B, decoder_channels, H/4, W/4] (approx)

class SmpClassificationHead(nn.Module):
    """Wrapper for SMP Classification Head."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = smp.base.ClassificationHead(
            in_channels=in_channels,
            classes=num_classes,
            pooling="avg",
            dropout=0.2,
            activation=None,
        )
        
    def forward(self, features: list):
        # Use the last feature map from encoder
        return self.head(features[-1])

import torch
import torch.nn as nn
from typing import List, Union

class RegressionHead(nn.Module):
    """Custom head for regression tasks (e.g., landmark detection).
    
    Args:
        in_channels (int): Number of input channels from the backbone feature map.
        num_points (int): Number of keypoints to predict.
        feature_index (int): Index of the feature map to use from the backbone output list.
            Default: -1 (use the last/largest-stride feature map).
    """
    def __init__(
        self,
        in_channels: int,
        num_points: int,
        feature_index: int = -1
    ):
        super().__init__()
        self.feature_index = feature_index
        self.num_points = num_points
        
        # Global average pooling + linear projection
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),  # flatten all but batch dim
            nn.Linear(in_channels, num_points * 2)
        )
        
        # Optional: better initialization for regression
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features (List[Tensor]): List of feature maps from backbone.
                Each tensor has shape (B, C_i, H_i, W_i).
        Returns:
            Tensor: Predicted coordinates of shape (B, num_points * 2),
                    where each pair is (x, y) normalized or in pixel space.
        """
        x = features[self.feature_index]  # more explicit than [-1]
        return self.head(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: 全局平均池化 → (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 压缩
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 恢复
            nn.Sigmoid()  # 生成通道权重 ∈ [0,1]
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)          # (B, C)
        y = self.fc(y).view(b, c, 1, 1)          # (B, C, 1, 1)
        return x * y.expand_as(x)                # 逐通道加权

class FPNGridDetectionHead(nn.Module):
    def __init__(self, fpn_out_channels: int, num_classes: int = 1, num_anchors: int = 1):
        super().__init__()
        mid_channels = fpn_out_channels // 2
        num_outputs = 4
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(fpn_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            # 添加注意力机制
            SELayer(mid_channels),  # 假设已定义SELayer
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            SELayer(mid_channels),  # 再次添加注意力机制
            nn.Conv2d(mid_channels, num_outputs, kernel_size=1)
        )

    def forward(self, fpn_features: torch.Tensor):
        predictions_map = self.conv_block(fpn_features)
        predictions_map = torch.sigmoid(predictions_map)
        # predictions_map[:, :4] = torch.sigmoid(predictions_map[:, :4])
        return predictions_map


# ====================================================================
# --- 2. Multi-Task Model Factory ---
# ====================================================================

class MultiTaskModelFactory(nn.Module):
    def __init__(self, encoder_name: str, encoder_weights: str, task_configs: List[Dict]):
        super().__init__()
        
        print(f"Initializing timm backbone: {encoder_name}")
        self.backbone = timm.create_model(
            encoder_name,
            features_only=True,
            pretrained=True,
            out_indices=(0, 1, 2, 3)  # ConvNeXt-Base has 5 stages → indices 0~4
        )

        self.backbone_1 = timm.create_model(
            encoder_name,
            features_only=True,
            pretrained=True,
            out_indices=(0, 1, 2, 3)  # ConvNeXt-Base has 5 stages → indices 0~4
        )
        
        # Get feature channels from timm model
        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384)
            feats = self.backbone(dummy)
            self.encoder_channels = [f.shape[1] for f in feats]
        print(f"Backbone output channels: {self.encoder_channels}")

        
        # Initialize shared encoder
        print(f"Initializing shared encoder: {encoder_name}")

        # self.fpn_decoder = temp_fpn_model.decoder
        self.fpn_decoder = TimmFPNDecoder(self.encoder_channels, decoder_channels=384)
        self.fpn_out_channels = self.fpn_decoder.out_channels
        
        # Initialize task heads
        self.heads = nn.ModuleDict()
        
        print(f"Creating heads for {len(task_configs)} tasks...")
        for config in task_configs:
            task_id = config['task_id']
            task_name = config['task_name']
            num_classes = config['num_classes']
            
            head_module = None
            if task_name == 'segmentation':
                head_module = smp.base.SegmentationHead(
                    in_channels=self.fpn_out_channels, 
                    out_channels=num_classes, 
                    kernel_size=1,
                    upsampling=4 
                )

            elif task_name == 'classification':
                head_module = SmpClassificationHead(
                    in_channels=self.encoder_channels[-1],
                    num_classes=num_classes
                )

            elif task_name == 'Regression':
                num_points = config['num_classes']
                head_module = RegressionHead(
                    in_channels=self.encoder_channels[-1],
                    num_points=num_points
                )

            elif task_name == 'detection':
                head_module = FPNGridDetectionHead(
                    fpn_out_channels=self.fpn_out_channels,
                    num_classes = num_classes
                )

            if head_module:
                self.heads[task_id] = head_module
            else:
                print(f"Warning: Unknown task type '{task_name}' for {task_id}")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found.")

        task_config = next((item for item in TASK_CONFIGURATIONS if item["task_id"] == task_id), None)
        task_name = task_config['task_name'] if task_config else None
        
        if task_name == 'detection':
            features = self.backbone(x)  # List[Tensor]: [C1, C2, C3, C4, C5]
        else:
            features = self.backbone_1(x)
        
        # Route features based on task type
        if task_name == 'segmentation':
            # Use FPN features for dense prediction tasks
            fpn_features = self.fpn_decoder(features)
            output = self.heads[task_id](fpn_features)
        elif task_name == 'detection':
            det_fpn_features = self.fpn_decoder(features)
            output = self.heads[task_id](det_fpn_features)
        else: 
            # Use encoder features directly for global prediction tasks
            output = self.heads[task_id](features)
            
        return output

from collections import OrderedDict

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_breakdown(model):
    """Print detailed parameter count for each major component."""
    print("=" * 60)
    print("MODEL PARAMETER BREAKDOWN")
    print("=" * 60)
    
    # 1. Backbone 1 (used for detection)
    backbone1_params = count_parameters(model.backbone)
    print(f"Backbone (self.backbone)        : {backbone1_params / 1e6:6.2f} M")

    # 2. Backbone 2 (used for cls/seg/reg)
    backbone2_params = count_parameters(model.backbone_1)
    print(f"Backbone (self.backbone_1)      : {backbone2_params / 1e6:6.2f} M")

    # 3. Shared FPN Decoder
    fpn_params = count_parameters(model.fpn_decoder)
    print(f"FPN Decoder                     : {fpn_params / 1e6:6.2f} M")

    # 4. Task Heads breakdown
    head_params_total = 0
    head_breakdown = OrderedDict()
    
    for task_id, head in model.heads.items():
        # Find task name from TASK_CONFIGURATIONS
        task_config = next((c for c in TASK_CONFIGURATIONS if c['task_id'] == task_id), None)
        task_name = task_config['task_name'] if task_config else 'unknown'
        
        p = count_parameters(head)
        head_params_total += p
        head_breakdown[task_id] = {
            'params': p,
            'task_type': task_name,
            'num_classes': task_config['num_classes'] if task_config else 0
        }
    
    print(f"\nTask Heads (detailed):")
    cls_total = seg_total = reg_total = det_total = 0
    for tid, info in head_breakdown.items():
        print(f"  {tid:<30} : {info['params'] / 1e6:6.3f} M ({info['task_type']})")
        if info['task_type'] == 'classification':
            cls_total += info['params']
        elif info['task_type'] == 'segmentation':
            seg_total += info['params']
        elif info['task_type'] == 'Regression':
            reg_total += info['params']
        elif info['task_type'] == 'detection':
            det_total += info['params']
    
    print(f"\nHead Type Totals:")
    print(f"  Classification heads           : {cls_total / 1e6:6.3f} M")
    print(f"  Segmentation heads             : {seg_total / 1e6:6.3f} M")
    print(f"  Regression heads               : {reg_total / 1e6:6.3f} M")
    print(f"  Detection heads                : {det_total / 1e6:6.3f} M")
    print(f"  → All Task Heads Total         : {head_params_total / 1e6:6.3f} M")

    # 5. Grand Total
    total_params = backbone1_params + backbone2_params + fpn_params + head_params_total
    print("-" * 60)
    print(f"TOTAL TRAINABLE PARAMETERS       : {total_params / 1e6:6.2f} M")
    print("=" * 60)

# Example usage

if __name__ == '__main__':
    model = MultiTaskModelFactory(
        encoder_name='convnext_base.fb_in22k_ft_in1k_384',
        encoder_weights='imagenet',
        task_configs=TASK_CONFIGURATIONS
    )
    print_parameter_breakdown(model)
    print("\n--- Forward Pass Test ---")
    dummy_image_batch = torch.randn(2, 3, 384, 384) # Reduced batch size for test

    # Test specific tasks
    test_tasks = ['cardiac_multi', 'fetal_plane_cls', 'FUGC', 'thyroid_nodule_det']
    
    for t_id in test_tasks:
        try:
            out = model(dummy_image_batch, task_id=t_id)
            print(f"Task: {t_id:<25} | Output Shape: {out.shape}")
        except Exception as e:
            print(f"Task: {t_id:<25} | Error: {e}")

    from thop import profile
    import torch
    
    model.eval()  # 切换到 eval 模式（避免 dropout 等影响）
    dummy_input = torch.randn(1, 3, 384, 384)
    
    # --- 1. 分割任务（使用 backbone_1 + FPN）---
    flops_seg, params_seg = profile(
        model,
        inputs=(dummy_input, 'cardiac_multi'),  # segmentation task_id
        verbose=False
    )
    print(f"Segmentation: {flops_seg / 1e9:.2f} GFLOPs")
    
    # --- 2. 检测任务（使用 backbone + FPN）---
    flops_det, params_det = profile(
        model,
        inputs=(dummy_input, 'thyroid_nodule_det'),
        verbose=False
    )
    print(f"Detection: {flops_det / 1e9:.2f} GFLOPs")
    
    # --- 3. 分类任务（使用 backbone_1 + GAP，无 FPN）---
    flops_cls, params_cls = profile(
        model,
        inputs=(dummy_input, 'fetal_plane_cls'),
        verbose=False
    )
    print(f"Classification: {flops_cls / 1e9:.2f} GFLOPs")
    
    # --- 4. 回归任务（同分类）---
    flops_reg, params_reg = profile(
        model,
        inputs=(dummy_input, 'FUGC'),
        verbose=False
    )
    print(f"Regression: {flops_reg / 1e9:.2f} GFLOPs")

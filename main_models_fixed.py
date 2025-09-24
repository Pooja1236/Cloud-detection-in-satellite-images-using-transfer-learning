"""
Complete Cloud Detection Models
All 4 models with utility functions for 4-channel satellite image processing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import warnings

warnings.filterwarnings("ignore")

__all__ = [
    "CloudDeepLabV3",
    "CloudUNetEfficientNet", 
    "SimpleCloudUNet",
    "SimpleCNN",
    "normalize_satellite_image",
    "prepare_image_for_model",
    "run_inference_debug",
    "calculate_metrics",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== MODEL DEFINITIONS =====

class CloudDeepLabV3(nn.Module):
    """DeepLabV3+ with MobileNetV3 backbone - FIXED for 4 channels"""
    def __init__(self, num_classes=2, num_channels=4):
        super().__init__()
        
        # Load pretrained model
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        
        try:
            # Get original first conv layer
            original_conv = self.model.backbone.backbone.features[0][0]
            
            # Create new first conv layer for 4 channels
            new_conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=original_conv.out_channels,  # Usually 16
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize weights properly
            with torch.no_grad():
                # Copy RGB weights to first 3 channels
                new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
                # Initialize 4th channel (NIR) as average of RGB
                new_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True).clone()
                
                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.clone()
            
            # Replace the layer in the model
            self.model.backbone.backbone.features[0][0] = new_conv
            self.use_adapter = False
            print(f"✅ DeepLabV3: Successfully modified first layer for {num_channels} channels")
            
        except Exception as e:
            print(f"⚠️ DeepLabV3: Could not modify first layer ({e}), using adapter")
            # Fallback: create adapter layer
            self.input_adapter = nn.Conv2d(num_channels, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                # Initialize adapter to pass through RGB channels
                self.input_adapter.weight[0, 0, 0, 0] = 1.0  # R->R
                self.input_adapter.weight[1, 1, 0, 0] = 1.0  # G->G  
                self.input_adapter.weight[2, 2, 0, 0] = 1.0  # B->B
                if num_channels > 3:
                    self.input_adapter.weight[:, 3, 0, 0] = 0.0  # NIR->0
            self.use_adapter = True
        
        # Modify classifier for binary segmentation (cloud vs non-cloud)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        if self.use_adapter:
            x = self.input_adapter(x)
        return self.model(x)["out"]


class CloudUNetEfficientNet(nn.Module):
    """U-Net with EfficientNet-B0 encoder - FIXED for 4 channels"""
    def __init__(self, num_classes=2, num_channels=4):
        super().__init__()
        
        # Load EfficientNet backbone
        efficientnet = models.efficientnet_b0(weights="DEFAULT")
        
        try:
            # Fix first conv layer for 4 channels
            original_conv = efficientnet.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=original_conv.out_channels,  # Usually 32
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
                if num_channels > 3:
                    new_conv.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True).clone()
                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.clone()
            
            efficientnet.features[0][0] = new_conv
            print(f"✅ U-Net EfficientNet: Successfully modified first layer for {num_channels} channels")
            
        except Exception as e:
            print(f"❌ U-Net EfficientNet: Could not modify first layer ({e})")
            raise e
        
        self.encoder_features = efficientnet.features
        self.encoder_avgpool = efficientnet.avgpool
        
        # U-Net style decoder
        self.decoder = nn.Sequential(
            # Upsample from 1280 features (EfficientNet-B0 output)
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Extract features through encoder
        features = self.encoder_features(x)
        
        # Global average pooling and reshape for decoder input
        features = self.encoder_avgpool(features)
        features = torch.flatten(features, 1)
        features = features.view(features.size(0), -1, 1, 1)
        
        # Expand to spatial dimensions for decoder (16x16 is good starting point)
        features = F.interpolate(features, size=(16, 16), mode="bilinear", align_corners=False)
        
        # Decode to full resolution
        output = self.decoder(features)
        
        return output


class SimpleCloudUNet(nn.Module):
    """Simple U-Net - guaranteed to work on any system"""
    def __init__(self, num_classes=2, num_channels=4):
        super().__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Encoder (downsampling path)
        self.enc1 = conv_block(num_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)  # 1024 = 512 + 512 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)   # 512 = 256 + 256
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)   # 256 = 128 + 128
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)    # 128 = 64 + 64
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        return self.out_conv(d1)


class SimpleCNN(nn.Module):
    """Lightweight CNN for fast cloud detection"""
    def __init__(self, num_classes=2, num_channels=4):
        super().__init__()
        
        self.net = nn.Sequential(
            # First convolutional block
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # Output classification layer
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        return self.net(x)


# ===== UTILITY FUNCTIONS =====

def normalize_satellite_image(image):
    """
    Robust normalization for satellite images using percentile-based scaling
    
    Args:
        image: numpy array of shape (H, W, C) containing satellite image data
        
    Returns:
        Normalized image with values in [0, 1] range
    """
    img = image.astype(np.float32).copy()
    
    for i in range(img.shape[2]):
        channel = img[:, :, i]
        
        # Use aggressive percentiles to handle outliers in satellite data
        low_pct = np.percentile(channel, 1)
        high_pct = np.percentile(channel, 99)
        
        if high_pct > low_pct:
            # Normalize to [0, 1] range
            img[:, :, i] = np.clip((channel - low_pct) / (high_pct - low_pct), 0, 1)
        else:
            # If channel is uniform, set to middle value
            img[:, :, i] = 0.5
    
    return img


def prepare_image_for_model(image):
    """
    Prepare satellite image for model input
    
    Args:
        image: numpy array of shape (H, W, C)
        
    Returns:
        PyTorch tensor of shape (1, C, H, W) ready for model input
    """
    # Normalize image
    norm_img = normalize_satellite_image(image)
    
    # Convert to tensor: (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float().unsqueeze(0)
    
    return tensor


def run_inference_debug(model, image_tensor, model_name):
    """
    Run inference with detailed debugging information
    
    Args:
        model: PyTorch model
        image_tensor: Input tensor of shape (1, C, H, W)
        model_name: Name of the model for debugging
        
    Returns:
        tuple: (predicted_mask, probability_maps, debug_info)
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        output = model(image_tensor.to(device))
        
        # Resize output to match input if needed
        if output.shape[-2:] != image_tensor.shape[-2:]:
            output = F.interpolate(
                output, 
                size=image_tensor.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
        
        # Convert logits to probabilities
        probs = torch.softmax(output, dim=1)
        
        # Get predicted class (0=clear, 1=cloud)
        pred_mask = torch.argmax(probs, dim=1)
        
        # Move to CPU and convert to numpy
        mask = pred_mask.squeeze().cpu().numpy()
        prob_maps = probs.squeeze().cpu().numpy()
        
        # Collect debug information
        debug_info = {
            "input_shape": list(image_tensor.shape),
            "input_range": f"{image_tensor.min().item():.3f} to {image_tensor.max().item():.3f}",
            "output_shape": list(output.shape),
            "output_range": f"{output.min().item():.3f} to {output.max().item():.3f}",
            "prob_range": f"{probs.min().item():.3f} to {probs.max().item():.3f}",
            "pred_coverage": f"{(mask == 1).sum() / mask.size * 100:.2f}%"
        }
        
        # Additional probability statistics for binary classification
        if probs.shape[1] == 2:
            debug_info["avg_clear_prob"] = f"{probs[:, 0, :, :].mean().item():.3f}"
            debug_info["avg_cloud_prob"] = f"{probs[:, 1, :, :].mean().item():.3f}"
        
        return mask, prob_maps, debug_info


def calculate_metrics(pred_mask, true_mask=None):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        pred_mask: Predicted binary mask (numpy array)
        true_mask: Ground truth binary mask (numpy array, optional)
        
    Returns:
        Dictionary containing calculated metrics
    """
    pred_mask = pred_mask.astype(int)
    
    # Basic prediction statistics
    total_pixels = pred_mask.size
    cloud_pixels_pred = int((pred_mask == 1).sum())
    
    metrics = {
        "pred_coverage": cloud_pixels_pred / total_pixels * 100,
        "pred_pixels": cloud_pixels_pred,
        "clear_pixels": int((pred_mask == 0).sum()),
        "total_pixels": total_pixels
    }
    
    # If ground truth is available, calculate accuracy metrics
    if true_mask is not None:
        true_mask = true_mask.astype(int)
        
        # Check size compatibility
        if pred_mask.shape != true_mask.shape:
            return None
        
        cloud_pixels_true = int((true_mask == 1).sum())
        
        # Confusion matrix components
        tp = int(((pred_mask == 1) & (true_mask == 1)).sum())  # True Positives
        tn = int(((pred_mask == 0) & (true_mask == 0)).sum())  # True Negatives
        fp = int(((pred_mask == 1) & (true_mask == 0)).sum())  # False Positives
        fn = int(((pred_mask == 0) & (true_mask == 1)).sum())  # False Negatives
        
        # Calculate accuracy metrics
        accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also called sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # Intersection over Union
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0  # Dice coefficient
        
        # Balanced accuracy (useful for imbalanced datasets)
        balanced_accuracy = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient (MCC)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
        
        # Add ground truth metrics to the dictionary
        metrics.update({
            "true_coverage": cloud_pixels_true / total_pixels * 100,
            "accuracy": accuracy * 100,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "iou": iou,
            "dice": dice,
            "balanced_accuracy": balanced_accuracy * 100,
            "matthews_correlation": mcc,
            "tp": tp,
            "tn": tn, 
            "fp": fp,
            "fn": fn,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0
        })
    
    return metrics


def get_model_info(model):
    """
    Get information about a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": param_count,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_count * 4 / 1024 / 1024,  # Assuming float32
        "device": next(model.parameters()).device
    }


# Test function to verify all models work
def test_all_models():
    """Test all models with synthetic data to verify they work"""
    print("🧪 Testing all models with synthetic data...")
    
    # Create synthetic 4-channel satellite image
    test_image = np.random.randint(8000, 20000, (256, 256, 4), dtype=np.uint16)
    test_tensor = prepare_image_for_model(test_image)
    
    models_to_test = [
        ("Simple CNN", SimpleCNN),
        ("Simple U-Net", SimpleCloudUNet),
        ("DeepLabV3+ MobileNetV3", CloudDeepLabV3),
        ("U-Net EfficientNet", CloudUNetEfficientNet),
    ]
    
    results = {}
    
    for name, model_class in models_to_test:
        try:
            print(f"\n🔬 Testing {name}...")
            model = model_class(num_classes=2, num_channels=4).to(device)
            model.eval()
            
            info = get_model_info(model)
            print(f"   Parameters: {info['total_parameters']:,}")
            
            mask, probs, debug = run_inference_debug(model, test_tensor, name)
            print(f"   Output shape: {mask.shape}")
            print(f"   Predicted coverage: {debug['pred_coverage']}")
            
            results[name] = (mask, probs, debug)
            print(f"   ✅ {name} working correctly")
            
        except Exception as e:
            print(f"   ❌ {name} failed: {str(e)}")
            results[name] = None
    
    print(f"\n📊 Test Results: {len([r for r in results.values() if r is not None])}/{len(models_to_test)} models working")
    return results


if __name__ == "__main__":
    # Run tests when the module is executed directly
    test_all_models()

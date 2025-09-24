import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch

MM = importlib.import_module("main_models_fixed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_synth():
    np.random.seed(42)
    H, W, C = 512, 512, 4
    img = np.random.randint(8000, 20000, (H, W, C), dtype=np.uint16)
    centers = [(150, 200), (300, 350), (100, 400), (400, 100)]
    sizes = [4000, 3000, 2500, 3500]
    for (cx, cy), s in zip(centers, sizes):
        y, x = np.ogrid[:H, :W]
        m = (x - cx) ** 2 + (y - cy) ** 2 < s
        img[m] = np.random.randint(25000, 40000, (m.sum(), C))
    gt = np.zeros((H, W), dtype=np.uint8)
    for (cx, cy), s in zip(centers, sizes):
        y, x = np.ogrid[:H, :W]
        m = (x - cx) ** 2 + (y - cy) ** 2 < s * 0.8
        gt[m] = 1
    return img, gt

def main():
    print("🛰️ Cloud Detection System Test (Fixed)")
    print("=" * 50)
    print(f"Device: {device}")

    img, gt = create_synth()
    print(f"Image: {img.shape}, GT: {gt.shape}, GT Cloud: {(gt.sum()/gt.size)*100:.2f}%")

    tried = [
        ("Simple U-Net", lambda: MM.SimpleCloudUNet(num_classes=2, num_channels=4)),
        ("Simple CNN", lambda: MM.SimpleCNN(num_classes=2, num_channels=4)),
        ("DeepLabV3+ MobileNetV3", lambda: MM.CloudDeepLabV3(num_classes=2, num_channels=4)),
        ("U-Net EfficientNet", lambda: MM.CloudUNetEfficientNet(num_classes=2, num_channels=4)),
    ]

    results = {}
    for name, ctor in tried:
        try:
            model = ctor().to(device).eval()
            t = MM.prepare_image_for_model(img)
            mask, probs = MM.run_inference(model, t)
            mets = MM.calculate_metrics(mask, gt)
            print(f"\n{name}: Coverage {mets['cloud_coverage_percent']:.2f}%  Acc {mets.get('accuracy', 0):.2f}%  F1 {mets.get('f1_score', 0):.3f}")
            results[name] = (mask, mets)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(results) >= 2:
                break
        except Exception as e:
            print(f"Skip {name}: {e}")

    if not results:
        print("\n❌ No models succeeded. Ensure main_models_fixed.py is present.")
        return

    # Visualization
    cols = max(3, len(results) + 1)
    fig, axes = plt.subplots(2, cols, figsize=(20, 10))
    rgb = MM.normalize_satellite_image(img[:, :, :3])
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("Original (RGB)"); axes[0, 0].axis("off")
    axes[1, 0].imshow(gt, cmap="gray"); axes[1, 0].set_title("Ground Truth"); axes[1, 0].axis("off")

    for i, (name, (mask, mets)) in enumerate(results.items(), start=1):
        axes[0, i].imshow(mask, cmap="gray"); axes[0, i].set_title(f"{name}\nCov {mets['cloud_coverage_percent']:.1f}%"); axes[0, i].axis("off")
        over = rgb.copy(); over[mask == 1] = [1, 0, 0]
        axes[1, i].imshow(over); axes[1, i].set_title(f"{name} Overlay\nAcc {mets.get('accuracy', 0):.1f}%"); axes[1, i].axis("off")

    for j in range(len(results) + 1, cols):
        axes[0, j].axis("off"); axes[1, j].axis("off")

    plt.tight_layout()
    plt.savefig("cloud_detection_test_results.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\n✅ Saved: cloud_detection_test_results.png")

if __name__ == "__main__":
    main()

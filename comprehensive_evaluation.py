import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score
import torch

# Robust import of model module
MM = None
try:
    MM = importlib.import_module("main_models_fixed")
except Exception as e:
    raise RuntimeError("Could not import main_models_fixed.py. Ensure it is in the same folder.") from e

# Shorthand
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_comprehensive_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    total = pred_mask.size
    cloud_pred = int((pred_mask == 1).sum())
    cloud_true = int((true_mask == 1).sum())

    tn = int(((pred_mask == 0) & (true_mask == 0)).sum())
    fp = int(((pred_mask == 1) & (true_mask == 0)).sum())
    fn = int(((pred_mask == 0) & (true_mask == 1)).sum())
    tp = int(((pred_mask == 1) & (true_mask == 1)).sum())

    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = jaccard_score(true_flat, pred_flat, average="binary", pos_label=1) if (cloud_true + cloud_pred) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    bal_acc = (rec + spec) / 2.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return {
        "pred_cloud_coverage_percent": cloud_pred / total * 100.0,
        "true_cloud_coverage_percent": cloud_true / total * 100.0,
        "coverage_difference_percent": abs(cloud_pred - cloud_true) / total * 100.0,
        "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1_score": f1,
        "jaccard_index": iou, "dice_coefficient": dice, "false_positive_rate": fpr,
        "false_negative_rate": fnr, "balanced_accuracy": bal_acc, "matthews_correlation": mcc,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=int),
    }

def create_comprehensive_visualization(image, true_mask, pred_mask, probs, model_name, metrics):
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])

    rgb = MM.normalize_satellite_image(image[:, :, :3])

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(rgb); ax1.set_title("Original (RGB)"); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 1])
    if image.shape[2] > 3:
        nir = MM.normalize_satellite_image(image[:, :, 3:4]).squeeze()
        ax2.imshow(nir, cmap="gray"); ax2.set_title("NIR"); ax2.axis("off")
    else:
        ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(true_mask, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax3.set_title(f"Ground Truth – Coverage: {metrics['true_cloud_coverage_percent']:.1f}%"); ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3]); ax4.imshow(pred_mask, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax4.set_title(f"{model_name} – Coverage: {metrics['pred_cloud_coverage_percent']:.1f}%"); ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 0])
    cloud_probs = None
    if probs is not None:
        if probs.ndim == 3:
            cloud_probs = probs[1] if probs.shape[0] > 1 else probs[0]
        else:
            cloud_probs = probs
    if cloud_probs is None:
        cloud_probs = np.zeros_like(pred_mask, dtype=float)
    im5 = ax5.imshow(cloud_probs, cmap="hot", vmin=0, vmax=1)
    ax5.set_title("Cloud Probability"); ax5.axis("off"); plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 1])
    diff = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=float)
    diff[(pred_mask == 1) & (true_mask == 1)] = [1, 1, 1]  # TP white
    diff[(pred_mask == 0) & (true_mask == 0)] = [0, 0, 0]  # TN black
    diff[(pred_mask == 1) & (true_mask == 0)] = [1, 0, 0]  # FP red
    diff[(pred_mask == 0) & (true_mask == 1)] = [0, 0, 1]  # FN blue
    ax6.imshow(diff); ax6.set_title("Error Map (Red=FP, Blue=FN, White=TP)"); ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 2])
    overlay = rgb.copy()
    overlay[pred_mask == 1] = [1, 0, 0]
    overlay[true_mask == 1] = overlay[true_mask == 1] * 0.5 + np.array([0, 1, 0]) * 0.5
    ax7.imshow(overlay); ax7.set_title("Overlay: Red=Pred, Green=True"); ax7.axis("off")

    ax8 = fig.add_subplot(gs[1, 3])
    side = np.hstack([true_mask, pred_mask])
    ax8.imshow(side, cmap="RdYlBu_r", vmin=0, vmax=1); ax8.set_title("True | Pred"); ax8.axvline(x=true_mask.shape[1]-0.5, color="w", lw=2); ax8.axis("off")

    ax9 = fig.add_subplot(gs[2, 0])
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax9,
                xticklabels=["Clear", "Cloud"], yticklabels=["Clear", "Cloud"])
    ax9.set_title("Confusion Matrix"); ax9.set_xlabel("Predicted"); ax9.set_ylabel("Actual")

    ax10 = fig.add_subplot(gs[2, 1:3]); ax10.axis("off")
    text = (
        f"ACCURACY METRICS\n"
        f"Accuracy: {metrics['accuracy']:.3f}\nBalanced Acc: {metrics['balanced_accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\nRecall: {metrics['recall']:.3f}\nSpecificity: {metrics['specificity']:.3f}\nF1: {metrics['f1_score']:.3f}\n\n"
        f"SIMILARITY\nIoU: {metrics['jaccard_index']:.3f}\nDice: {metrics['dice_coefficient']:.3f}\nMCC: {metrics['matthews_correlation']:.3f}\n\n"
        f"ERROR RATES\nFPR: {metrics['false_positive_rate']:.3f}\nFNR: {metrics['false_negative_rate']:.3f}\nCoverage Δ: {metrics['coverage_difference_percent']:.1f}%"
    )
    ax10.text(0.05, 0.95, text, va="top", family="monospace", bbox=dict(boxstyle="round", fc="lightgray", alpha=0.8))

    ax11 = fig.add_subplot(gs[2, 3]); ax11.axis("off")
    text2 = (
        f"PIXEL COUNTS\nTP: {metrics['true_positives']:,}\nFP: {metrics['false_positives']:,}\n"
        f"TN: {metrics['true_negatives']:,}\nFN: {metrics['false_negatives']:,}\n\n"
        f"COVERAGE\nTrue: {metrics['true_cloud_coverage_percent']:.2f}%\nPred: {metrics['pred_cloud_coverage_percent']:.2f}%\n"
        f"Model: {model_name}\nTotal: {pred_mask.size:,}"
    )
    ax11.text(0.05, 0.95, text2, va="top", family="monospace", bbox=dict(boxstyle="round", fc="lightblue", alpha=0.8))

    ax12 = fig.add_subplot(gs[3, :2])
    names = ["Acc", "Prec", "Rec", "F1", "IoU", "Dice"]
    vals = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['jaccard_index'], metrics['dice_coefficient']]
    bars = ax12.bar(names, vals, color=["skyblue", "lightgreen", "orange", "gold", "pink", "lightcoral"])
    ax12.set_ylim(0, 1); ax12.set_title("Performance Overview"); ax12.set_ylabel("Score")
    for b, v in zip(bars, vals): ax12.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", va="bottom")
    plt.setp(ax12.get_xticklabels(), rotation=0)

    ax13 = fig.add_subplot(gs[3, 2:])
    if cloud_probs is not None and np.any(cloud_probs > 0):
        ths = np.linspace(0, 1, 21)
        tprs, fprs = [], []
        for th in ths:
            th_pred = (cloud_probs > th).astype(int)
            m = calculate_comprehensive_metrics(th_pred, true_mask)
            tprs.append(m["recall"]); fprs.append(m["false_positive_rate"])
        ax13.plot(fprs, tprs, "b-o", ms=3); ax13.plot([0, 1], [0, 1], "r--", alpha=0.5)
        ax13.set_xlim(0, 1); ax13.set_ylim(0, 1); ax13.grid(True, alpha=0.3)
        ax13.set_title("ROC-like Curve"); ax13.set_xlabel("FPR"); ax13.set_ylabel("TPR")
        auc_approx = np.trapz(tprs, fprs); ax13.text(0.6, 0.2, f"AUC ≈ {abs(auc_approx):.3f}",
                                                     bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax13.text(0.5, 0.5, "No probability data", ha="center", va="center")
        ax13.set_title("ROC Analysis"); ax13.set_xlabel("FPR"); ax13.set_ylabel("TPR"); ax13.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_synthetic_satellite_data_with_labels():
    np.random.seed(42)
    H, W, C = 512, 512, 4
    image = np.random.randint(5000, 15000, (H, W, C), dtype=np.uint16)

    image[300:, :200] = np.random.randint(3000, 8000, (H-300, 200, C))
    image[200:300, 200:350] = np.random.randint(8000, 18000, (100, 150, C))
    image[350:, 350:] = np.random.randint(12000, 25000, (H-350, W-350, C))

    clouds = [
        {"center": (150, 200), "size": 4500, "intensity": (28000, 45000), "type": "cumulus"},
        {"center": (100, 400), "size": 3000, "intensity": (25000, 38000), "type": "stratocumulus"},
        {"center": (400, 150), "size": 2000, "intensity": (20000, 32000), "type": "cirrus"},
        {"center": (350, 350), "size": 5000, "intensity": (30000, 50000), "type": "cumulonimbus"},
        {"center": (250, 100), "size": 1800, "intensity": (22000, 35000), "type": "small_cumulus"},
    ]

    true_mask = np.zeros((H, W), dtype=np.uint8)
    for cl in clouds:
        cx, cy = cl["center"]; size = cl["size"]; a, b = cl["intensity"]
        y, x = np.ogrid[:H, :W]
        if cl["type"] == "cirrus":
            region = ((x - cx)**2 + (y - cy)**2 < size) & (np.random.random((H, W)) > 0.3)
        else:
            region = ((x - cx)**2 + (y - cy)**2 < size)
        image[region] = np.random.randint(a, b, (region.sum(), C))
        truth = ((x - cx)**2 + (y - cy)**2 < size * 0.85)
        true_mask[truth] = 1
        edge = ((x - cx)**2 + (y - cy)**2 < size * 1.2) & ~region
        image[edge] = (image[edge] + np.random.randint(15000, 25000, (edge.sum(), C))) // 2

    return image, true_mask

def test_model(model, model_name, image, true_mask):
    print(f"\n{'='*60}\nEVALUATION: {model_name}\n{'='*60}")
    t = MM.prepare_image_for_model(image)
    pred, probs = MM.run_inference(model, t)
    mets = calculate_comprehensive_metrics(pred, true_mask)

    print("\nPerformance:")
    print(f"Accuracy: {mets['accuracy']:.3f}  Precision: {mets['precision']:.3f}  Recall: {mets['recall']:.3f}  F1: {mets['f1_score']:.3f}")
    print(f"IoU: {mets['jaccard_index']:.3f}  Dice: {mets['dice_coefficient']:.3f}  MCC: {mets['matthews_correlation']:.3f}")
    print(f"Coverage - True: {mets['true_cloud_coverage_percent']:.2f}%  Pred: {mets['pred_cloud_coverage_percent']:.2f}%  Δ: {mets['coverage_difference_percent']:.2f}%")

    print("\nCreating visualization...")
    fig = create_comprehensive_visualization(image, true_mask, pred, probs, model_name, mets)
    out = f"comprehensive_evaluation_{model_name.replace(' ', '_').replace('+', 'plus').lower()}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    return pred, probs, mets, fig

def main():
    print("🛰️ COMPREHENSIVE CLOUD DETECTION EVALUATION")
    print("=" * 70)
    print("\n📊 Creating synthetic satellite data...")
    image, true_mask = create_synthetic_satellite_data_with_labels()
    print(f"Image: {image.shape}, Mask: {true_mask.shape}, GT Cloud: {(true_mask.sum()/true_mask.size)*100:.2f}%")
    print(f"Range: {image.min()}–{image.max()}  True Cloud Pixels: {int(true_mask.sum()):,}")

    print(f"🖥️ Device: {device}")
    results = {}

    to_try = [
        ("Simple U-Net", lambda: MM.SimpleCloudUNet(num_classes=2, num_channels=4)),
        ("Simple CNN", lambda: MM.SimpleCNN(num_classes=2, num_channels=4)),
        # Add heavier ones last
        ("DeepLabV3+ MobileNetV3", lambda: MM.CloudDeepLabV3(num_classes=2, num_channels=4)),
        ("U-Net EfficientNet", lambda: MM.CloudUNetEfficientNet(num_classes=2, num_channels=4)),
    ]

    for name, ctor in to_try:
        try:
            print(f"\n🤖 Loading {name}...")
            model = ctor().to(device).eval()
            pred, probs, mets, fig = test_model(model, name, image, true_mask)
            results[name] = dict(pred=pred, probs=probs, metrics=mets, fig=fig)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Failed: {name} → {e}")

    if not results:
        print("\n❌ No models succeeded. Ensure main_models_fixed.py is in the same folder and PyTorch/torchvision are installed.")
    else:
        print(f"\n✅ Completed with {len(results)} result(s).")

if __name__ == "__main__":
    main()

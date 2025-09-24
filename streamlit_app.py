"""
Complete Streamlit Cloud Detection Interface
Imports models from main_models_complete.py and provides full GUI
"""

import os
import sys
import warnings
import traceback

# ULTIMATE Windows fixes - MUST be first
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

# Set matplotlib backend to prevent GUI issues
plt.switch_backend('Agg')
torch.set_num_threads(1)

# Import our models
try:
    from main_models_fixed import (
        CloudDeepLabV3,
        CloudUNetEfficientNet,
        SimpleCloudUNet,
        SimpleCNN,
        normalize_satellite_image,
        prepare_image_for_model,
        run_inference_debug,
        calculate_metrics,
        get_model_info
    )
except ImportError as e:
    st.error(f"❌ Could not import models from main_models_complete.py: {e}")
    st.info("Make sure main_models_complete.py is in the same folder as this script")
    st.stop()

st.set_page_config(
    page_title="Complete Cloud Detection System", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== MAIN APP =====

st.title("🛰️ Complete Cloud Detection System")
st.markdown("### Professional cloud detection with 4 state-of-the-art models")

# Device and system info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.header("📊 System Information")
st.sidebar.write(f"**Device:** {device}")
st.sidebar.write(f"**PyTorch:** {torch.__version__}")
st.sidebar.write(f"**Platform:** {sys.platform}")

# Model selection
st.sidebar.header("🤖 Model Selection")
available_models = {
    "Simple U-Net": SimpleCloudUNet,
    "Simple CNN": SimpleCNN,
    "DeepLabV3+ MobileNetV3": CloudDeepLabV3,
    "U-Net EfficientNet": CloudUNetEfficientNet
}

selected_models = st.sidebar.multiselect(
    "Choose models to load:",
    options=list(available_models.keys()),
    default=["Simple U-Net", "Simple CNN"],
    help="Start with simple models if you encounter issues"
)

# Display options
st.sidebar.header("🔧 Display Options")
show_debug = st.sidebar.checkbox("Show debug information", value=True)
show_prob = st.sidebar.checkbox("Show probability maps", value=True)
show_overlay = st.sidebar.checkbox("Show overlay visualizations", value=True)
show_error_map = st.sidebar.checkbox("Show error analysis maps", value=True)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    max_image_size = st.slider("Max image dimension", 128, 1024, 512)
    resize_images = st.checkbox("Resize large images", value=True)

@st.cache_resource
def load_selected_models(model_names):
    """Load and cache selected models"""
    models = {}
    model_info = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, name in enumerate(model_names):
        try:
            status_text.text(f"Loading {name}...")
            progress_bar.progress((i + 1) / len(model_names))
            
            # Create model instance
            torch.set_num_threads(1)  # Windows compatibility
            model_class = available_models[name]
            model = model_class(num_classes=2, num_channels=4).to(device)
            model.eval()
            
            # Get model information
            info = get_model_info(model)
            
            models[name] = model
            model_info[name] = info
            
            st.sidebar.success(f"✅ {name}: {info['total_parameters']:,} params")
            
        except Exception as e:
            st.sidebar.error(f"❌ {name}: {str(e)[:100]}...")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return models, model_info

def load_numpy_file_safe(uploaded_file):
    """Safely load numpy files with comprehensive error handling"""
    # Skip macOS resource fork files
    if uploaded_file.name.startswith('._'):
        return None, "macOS resource fork file (skip these)"
    
    # Check file size
    if uploaded_file.size < 100:
        return None, f"File too small ({uploaded_file.size} bytes)"
    
    try:
        uploaded_file.seek(0)
        
        # Check numpy magic header
        magic = uploaded_file.read(6)
        uploaded_file.seek(0)
        
        if magic[:6] != b'\x93NUMPY':
            return None, "Not a valid numpy (.npy) file"
        
        # Try loading without pickle first (safer)
        try:
            data = np.load(uploaded_file, allow_pickle=False)
            return data, None
        except ValueError as e:
            if "pickled data" in str(e):
                # Try with pickle allowed
                uploaded_file.seek(0)
                data = np.load(uploaded_file, allow_pickle=True)
                
                # Handle object arrays
                if data.dtype == object:
                    if data.ndim == 0:
                        data = data.item()
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                
                return data, "⚠️ Loaded pickled data (less secure)"
            else:
                return None, f"Loading error: {str(e)[:100]}"
                
    except Exception as e:
        return None, f"Failed to load: {str(e)[:100]}"

def resize_image_if_needed(image, max_size):
    """Resize image if it's too large"""
    if not resize_images:
        return image, False
    
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        # Calculate new dimensions
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        
        # Resize each channel
        from scipy import ndimage
        resized = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
        for c in range(image.shape[2]):
            resized[:, :, c] = ndimage.zoom(image[:, :, c], (new_h/h, new_w/w), order=1)
        
        return resized, True
    
    return image, False

def create_error_analysis_map(pred_mask, true_mask):
    """Create color-coded error analysis map"""
    error_map = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    
    # True Positives (correctly detected clouds) - White
    error_map[(pred_mask == 1) & (true_mask == 1)] = [1, 1, 1]
    
    # True Negatives (correctly detected clear sky) - Black
    error_map[(pred_mask == 0) & (true_mask == 0)] = [0, 0, 0]
    
    # False Positives (predicted cloud, actually clear) - Red
    error_map[(pred_mask == 1) & (true_mask == 0)] = [1, 0, 0]
    
    # False Negatives (missed clouds) - Blue
    error_map[(pred_mask == 0) & (true_mask == 1)] = [0, 0, 1]
    
    return error_map

# ===== FILE UPLOAD INTERFACE =====

st.subheader("📁 Upload Files")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.write("**📡 Satellite Images (.npy)**")
    uploaded_images = st.file_uploader(
        "Upload 4-channel satellite images",
        type=["npy"],
        accept_multiple_files=True,
        key="images",
        help="Expected format: (Height, Width, 4) - RGB + NIR channels"
    )

with col_upload2:
    st.write("**🎯 Ground Truth Masks (.npy)**")
    uploaded_masks = st.file_uploader(
        "Upload binary cloud masks (optional)",
        type=["npy"],
        accept_multiple_files=True,
        key="masks",
        help="Binary masks: 0=clear sky, 1=cloud"
    )

# Sample data generator
if st.button("🧪 Generate Sample Data for Testing"):
    with st.spinner("Creating synthetic satellite data..."):
        np.random.seed(42)
        
        # Create realistic sample image
        sample_img = np.random.randint(8000, 20000, (512, 512, 4), dtype=np.uint16)
        sample_mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Add realistic cloud formations
        cloud_configs = [
            {"center": (150, 200), "size": 4000, "intensity": (28000, 45000)},
            {"center": (300, 350), "size": 3500, "intensity": (25000, 40000)},
            {"center": (100, 400), "size": 2800, "intensity": (30000, 48000)},
            {"center": (400, 100), "size": 2200, "intensity": (26000, 42000)},
        ]
        
        for cloud in cloud_configs:
            cx, cy = cloud["center"]
            size = cloud["size"]
            min_int, max_int = cloud["intensity"]
            
            y, x = np.ogrid[:512, :512]
            cloud_region = (x - cx)**2 + (y - cy)**2 < size
            
            # Add clouds to image
            sample_img[cloud_region] = np.random.randint(min_int, max_int, (cloud_region.sum(), 4))
            
            # Add to ground truth
            sample_mask[cloud_region] = 1
        
        # Save files
        np.save("sample_satellite_image.npy", sample_img)
        np.save("sample_ground_truth_mask.npy", sample_mask)
        
    st.success("✅ Created sample files: sample_satellite_image.npy and sample_ground_truth_mask.npy")
    st.info("🔄 Upload these files above to test the complete system!")

# ===== MAIN PROCESSING =====

if uploaded_images and selected_models:
    # Load models
    models, model_info = load_selected_models(selected_models)
    
    if not models:
        st.error("❌ No models could be loaded successfully")
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common issues:**
            - Try starting with "Simple U-Net" and "Simple CNN" only
            - Check PyTorch installation: `pip install torch torchvision`
            - For Windows: Set environment variable `KMP_DUPLICATE_LIB_OK=TRUE`
            - Restart Streamlit if models fail to load
            """)
        st.stop()
    
    st.success(f"✅ Successfully loaded {len(models)} model(s)")
    
    # Process each uploaded image
    for img_idx, img_file in enumerate(uploaded_images):
        st.markdown("---")
        st.header(f"🖼️ Analysis: {img_file.name}")
        
        # Load and validate image
        img, img_warning = load_numpy_file_safe(img_file)
        
        if img is None:
            st.error(f"❌ Failed to load {img_file.name}: {img_warning}")
            continue
        
        if img_warning:
            st.warning(img_warning)
        
        # Validate image format
        if img.ndim != 3:
            st.error(f"❌ Expected 3D array (H×W×C), got {img.ndim}D array with shape {img.shape}")
            continue
        
        if img.shape[2] not in [3, 4]:
            st.error(f"❌ Expected 3 or 4 channels, got {img.shape[2]} channels")
            continue
        
        # Handle 3-channel images by adding dummy NIR
        if img.shape[2] == 3:
            st.info("📝 Adding dummy NIR channel (copy of red band) for 4-channel compatibility")
            nir_channel = img[:, :, 0:1]
            img = np.concatenate([img, nir_channel], axis=2)
        
        # Resize if needed
        original_shape = img.shape
        img, was_resized = resize_image_if_needed(img, max_image_size)
        
        if was_resized:
            st.info(f"📏 Resized from {original_shape[:2]} to {img.shape[:2]} for processing")
        
        # Display image info
        st.success(f"✅ Image loaded: {img.shape} ({img.dtype})")
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Height", img.shape[0])
        with col_info2:
            st.metric("Width", img.shape[1])
        with col_info3:
            st.metric("Channels", img.shape[2])
        with col_info4:
            st.metric("Value Range", f"{img.min():,}–{img.max():,}")
        
        # Load corresponding ground truth mask
        true_mask = None
        if uploaded_masks and img_idx < len(uploaded_masks):
            mask_file = uploaded_masks[img_idx]
            mask_data, mask_warning = load_numpy_file_safe(mask_file)
            
            if mask_data is not None:
                if mask_warning:
                    st.warning(f"Mask: {mask_warning}")
                
                if mask_data.ndim == 2:
                    true_mask = mask_data.astype(int)
                    
                    # Resize mask if image was resized
                    if was_resized:
                        from scipy import ndimage
                        h_ratio = img.shape[0] / original_shape[0]
                        w_ratio = img.shape[1] / original_shape[1]
                        true_mask = ndimage.zoom(true_mask, (h_ratio, w_ratio), order=0)
                    
                    true_coverage = (true_mask == 1).sum() / true_mask.size * 100
                    st.success(f"✅ Ground truth loaded: {true_mask.shape} ({true_coverage:.2f}% clouds)")
                else:
                    st.warning(f"Expected 2D mask, got {mask_data.ndim}D array")
            else:
                st.warning(f"Could not load ground truth mask: {mask_warning}")
        
        # Display original data
        st.subheader("📷 Original Satellite Data")
        
        rgb = normalize_satellite_image(img[:, :, :3])
        nir = normalize_satellite_image(img[:, :, 3:4]).squeeze()
        
        display_cols = st.columns(3 if true_mask is not None else 2)
        
        with display_cols[0]:
            st.write("**RGB Composite**")
            fig_rgb, ax_rgb = plt.subplots(figsize=(6, 6))
            ax_rgb.imshow(rgb)
            ax_rgb.set_title("RGB Channels", fontsize=14)
            ax_rgb.axis("off")
            st.pyplot(fig_rgb)
            plt.close(fig_rgb)
        
        with display_cols[1]:
            st.write("**NIR Channel**")
            fig_nir, ax_nir = plt.subplots(figsize=(6, 6))
            ax_nir.imshow(nir, cmap="gray")
            ax_nir.set_title("Near-Infrared Band", fontsize=14)
            ax_nir.axis("off")
            st.pyplot(fig_nir)
            plt.close(fig_nir)
        
        if true_mask is not None:
            with display_cols[2]:
                st.write("**Ground Truth**")
                fig_gt, ax_gt = plt.subplots(figsize=(6, 6))
                ax_gt.imshow(true_mask, cmap="RdYlBu_r")
                ax_gt.set_title(f"True Cloud Mask\n{(true_mask==1).sum()/true_mask.size*100:.1f}% coverage", fontsize=14)
                ax_gt.axis("off")
                st.pyplot(fig_gt)
                plt.close(fig_gt)
        
        # Process with each selected model
        st.subheader("🤖 Model Inference Results")
        
        results = {}
        image_tensor = prepare_image_for_model(img)
        
        for model_name, model in models.items():
            st.write(f"### 🔬 {model_name}")
            
            try:
                with st.spinner(f"Running {model_name} inference..."):
                    pred_mask, prob_maps, debug_info = run_inference_debug(model, image_tensor, model_name)
                    metrics = calculate_metrics(pred_mask, true_mask)
                    results[model_name] = (pred_mask, prob_maps, metrics, debug_info)
                
                # Display debug information
                if show_debug:
                    with st.expander(f"🔍 Debug Information: {model_name}"):
                        col_debug1, col_debug2 = st.columns(2)
                        
                        with col_debug1:
                            for key, value in debug_info.items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        with col_debug2:
                            info = model_info[model_name]
                            st.write(f"**Parameters:** {info['total_parameters']:,}")
                            st.write(f"**Model Size:** {info['model_size_mb']:.1f} MB")
                            st.write(f"**Device:** {info['device']}")
                
                # Display visual results
                num_cols = 5 if (show_prob and show_overlay and show_error_map and true_mask is not None) else 4
                result_cols = st.columns(num_cols)
                
                col_idx = 0
                
                # Predicted mask
                with result_cols[col_idx]:
                    st.write("**Predicted Mask**")
                    fig_pred, ax_pred = plt.subplots(figsize=(4, 4))
                    ax_pred.imshow(pred_mask, cmap="RdYlBu_r")
                    ax_pred.set_title(f"Cloud Prediction\n{metrics['pred_coverage']:.1f}% coverage")
                    ax_pred.axis("off")
                    st.pyplot(fig_pred)
                    plt.close(fig_pred)
                col_idx += 1
                
                # Probability map
                if show_prob and prob_maps is not None:
                    with result_cols[col_idx]:
                        st.write("**Probability Map**")
                        cloud_probs = prob_maps[1] if (prob_maps.ndim == 3 and prob_maps.shape[0] > 1) else prob_maps
                        fig_prob, ax_prob = plt.subplots(figsize=(4, 4))
                        im = ax_prob.imshow(cloud_probs, cmap="hot", vmin=0, vmax=1)
                        ax_prob.set_title("Cloud Confidence")
                        ax_prob.axis("off")
                        plt.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)
                        st.pyplot(fig_prob)
                        plt.close(fig_prob)
                    col_idx += 1
                
                # Overlay visualization
                if show_overlay:
                    with result_cols[col_idx]:
                        st.write("**Overlay**")
                        overlay = rgb.copy()
                        overlay[pred_mask == 1] = [1, 0, 0]  # Red for predicted clouds
                        if true_mask is not None:
                            # Green tint for true clouds
                            overlay[true_mask == 1] = overlay[true_mask == 1] * 0.7 + np.array([0, 0.3, 0])
                        
                        fig_over, ax_over = plt.subplots(figsize=(4, 4))
                        ax_over.imshow(overlay)
                        title = "Red=Pred, Green=True" if true_mask is not None else "Red=Predicted"
                        ax_over.set_title(title)
                        ax_over.axis("off")
                        st.pyplot(fig_over)
                        plt.close(fig_over)
                    col_idx += 1
                
                # Error analysis map
                if show_error_map and true_mask is not None:
                    with result_cols[col_idx]:
                        st.write("**Error Analysis**")
                        error_map = create_error_analysis_map(pred_mask, true_mask)
                        
                        fig_err, ax_err = plt.subplots(figsize=(4, 4))
                        ax_err.imshow(error_map)
                        ax_err.set_title("White=TP, Red=FP\nBlue=FN, Black=TN")
                        ax_err.axis("off")
                        st.pyplot(fig_err)
                        plt.close(fig_err)
                
                # Display performance metrics
                if true_mask is not None:
                    st.write("**📊 Performance Metrics**")
                    metric_cols = st.columns(6)
                    
                    with metric_cols[0]:
                        st.metric("🎯 Accuracy", f"{metrics['accuracy']:.1f}%")
                    with metric_cols[1]:
                        st.metric("📊 Precision", f"{metrics['precision']:.3f}")
                    with metric_cols[2]:
                        st.metric("🔍 Recall", f"{metrics['recall']:.3f}")
                    with metric_cols[3]:
                        st.metric("⚖️ F1 Score", f"{metrics['f1']:.3f}")
                    with metric_cols[4]:
                        st.metric("🔗 IoU", f"{metrics['iou']:.3f}")
                    with metric_cols[5]:
                        st.metric("🎲 Dice", f"{metrics['dice']:.3f}")
                else:
                    st.write("**📊 Prediction Statistics**")
                    stat_cols = st.columns(3)
                    
                    with stat_cols[0]:
                        st.metric("☁️ Cloud Coverage", f"{metrics['pred_coverage']:.2f}%")
                    with stat_cols[1]:
                        st.metric("📊 Cloud Pixels", f"{metrics['pred_pixels']:,}")
                    with stat_cols[2]:
                        st.metric("🌤️ Clear Pixels", f"{metrics['clear_pixels']:,}")
                
            except Exception as e:
                st.error(f"❌ Error with {model_name}: {str(e)}")
                if show_debug:
                    st.code(traceback.format_exc())
        
        # Model comparison summary
        if len(results) > 1:
            st.subheader("📊 Model Comparison Summary")
            
            comparison_data = []
            for name, (mask, _, metrics, _) in results.items():
                info = model_info[name]
                
                row = {
                    "Model": name,
                    "Parameters": f"{info['total_parameters']:,}",
                    "Size (MB)": f"{info['model_size_mb']:.1f}",
                    "Coverage (%)": f"{metrics['pred_coverage']:.2f}"
                }
                
                if true_mask is not None and 'accuracy' in metrics:
                    row.update({
                        "Accuracy (%)": f"{metrics['accuracy']:.1f}",
                        "F1": f"{metrics['f1']:.3f}",
                        "IoU": f"{metrics['iou']:.3f}",
                        "Dice": f"{metrics['dice']:.3f}"
                    })
                
                comparison_data.append(row)
            
            st.table(comparison_data)
            
            # Model agreement analysis
            if len(results) == 2:
                model_names = list(results.keys())
                mask1 = results[model_names[0]][0]
                mask2 = results[model_names[1]][0]
                agreement = (mask1 == mask2).sum() / mask1.size * 100
                
                st.metric("🤝 Model Agreement", f"{agreement:.1f}%")
                
                if agreement < 80:
                    st.warning("⚠️ Low model agreement - results may be uncertain")
                elif agreement > 95:
                    st.success("✅ High model agreement - results are consistent")

elif uploaded_images and not selected_models:
    st.warning("⚠️ Please select at least one model from the sidebar to begin analysis")

elif not uploaded_images:
    st.info("👆 Upload satellite image files (.npy format) to begin cloud detection analysis")
    
    # Instructions
    with st.expander("📖 Complete Usage Guide", expanded=True):
        st.markdown("""
        ## 🎯 How to Use This System
        
        ### Step 1: Model Selection
        - Choose models from the sidebar (start with Simple models for reliability)
        - **Simple U-Net**: Most reliable, always works
        - **Simple CNN**: Fastest, lightweight
        - **DeepLabV3+ MobileNetV3**: State-of-the-art, may require good hardware
        - **U-Net EfficientNet**: Modern architecture, balanced performance
        
        ### Step 2: Data Upload
        - **Images**: Upload 4-channel satellite images (.npy format)
        - **Masks**: (Optional) Upload ground truth masks for evaluation
        - Use the "Generate Sample Data" button to test the system
        
        ### Step 3: Analysis
        - View predictions, probability maps, and overlays
        - Compare multiple models side-by-side
        - Get detailed performance metrics with ground truth
        
        ## 📋 File Requirements
        
        ### Satellite Images
        - **Format**: NumPy .npy files
        - **Shape**: (Height, Width, 4) for 4-channel data
        - **Channels**: Usually RGB + NIR
        - **Size**: Typically > 1KB for real data
        
        ### Ground Truth Masks
        - **Format**: NumPy .npy files
        - **Shape**: (Height, Width) - same spatial size as images
        - **Values**: Binary (0=clear sky, 1=cloud)
        
        ## 🔧 Performance Tips
        
        - Start with smaller images (< 512×512) for faster processing
        - Use "Simple" models first to ensure system works
        - Enable debug info to understand model behavior
        - Resize large images automatically for performance
        
        ## 💡 Interpreting Results
        
        ### Metrics Explanation
        - **Accuracy**: Overall correctness (good > 90%)
        - **Precision**: Of predicted clouds, how many are real (good > 0.8)
        - **Recall**: Of real clouds, how many detected (good > 0.8)
        - **F1 Score**: Balanced precision/recall (good > 0.8)
        - **IoU**: Intersection over Union (good > 0.7)
        
        ### Visual Analysis
        - **Red overlay**: Model predictions
        - **Green overlay**: Ground truth (if available)
        - **Error map**: White=correct clouds, Red=false alarms, Blue=missed clouds
        """)
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        ## Common Issues & Solutions
        
        ### Windows PyTorch Crashes
        ```
        set KMP_DUPLICATE_LIB_OK=TRUE
        streamlit run streamlit_app_complete.py --server.fileWatcherType=poll
        ```
        
        ### Models Won't Load
        - Try only "Simple U-Net" and "Simple CNN" first
        - Check PyTorch installation: `pip install torch torchvision`
        - Restart Streamlit application
        
        ### File Upload Errors
        - Avoid uploading macOS resource files (starting with `._`)
        - Ensure files are actual .npy format
        - Check file isn't corrupted (should be > 1KB)
        
        ### Performance Issues
        - Reduce image size using the slider
        - Load fewer models simultaneously
        - Close other applications to free memory
        
        ### Getting "All Cloud" Predictions
        - This usually indicates normalization issues
        - Try the sample data first to test
        - Check your image value ranges (typical: 0-65535 for satellite data)
        """)

# Footer with system status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 System Status")
if selected_models and uploaded_images:
    st.sidebar.success("✅ Ready for analysis")
elif selected_models:
    st.sidebar.info("📁 Upload images to continue")
elif uploaded_images:
    st.sidebar.warning("🤖 Select models to continue")
else:
    st.sidebar.info("🚀 Ready to start")

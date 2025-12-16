import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from IPython.display import Image, display # Required for Colab display
# ==========================================
# 1. CUSTOM HOOK CLASS (Your Implementation)
# ==========================================

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, model, input, output):
        # Save features (We detach here to save memory during inference)
        if isinstance(output, (list, tuple)):
            self.features = output[0].detach()
        else:
            self.features = output.detach()

    def close(self):
        self.hook.remove()

# ==========================================
# 2. FEATURE EXTRACTION ENGINE
# ==========================================

def extract_dataset_features(model_folder, inputs_list, gt_list, target_stage=2, chk_name='checkpoint_final.pth', max_voxels=100000):
    """
    Iterates over multiple cases, extracts features, and aggregates them.
    max_voxels: Limit total points to prevent RAM explosion during t-SNE.
    """
    print(f"\n--- Loading Model from: {os.path.basename(model_folder)} ---")
    
    # 1. Initialize Predictor (Done once)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False
    )
    
    predictor.initialize_from_trained_model_folder(
        model_folder, 
        use_folds=('0',), 
        checkpoint_name=chk_name
    )

    # Instantiate Preprocessor
    print("Initializing Preprocessor...")
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
    
    # Containers for aggregated data
    all_X = []
    all_y = []
    
    predictor.network = predictor.network.cuda()
    # Hook Setup
    try:
        layer = predictor.network.decoder.stages[target_stage]
        conv_to_hook = getattr(layer, "conv", layer)
        print(f"Hooking into Decoder Stage {target_stage}: {conv_to_hook}")
    except IndexError:
        print(f"Error: Stage {target_stage} does not exist.")
        return None, None

    # 2. Loop over cases
    print(f"Processing {len(inputs_list)} cases...")
    
    for i, (case_img, case_gt) in enumerate(zip(inputs_list, gt_list)):
        print(f"  [{i+1}/{len(inputs_list)}] Processing: {os.path.basename(case_img[0])}")
        
        # A. Preprocess
        data, seg, data_properties = preprocessor.run_case(
            case_img, 
            seg_file=case_gt, 
            plans_manager=predictor.plans_manager, 
            configuration_manager=predictor.configuration_manager,
            dataset_json=predictor.dataset_json
        )

        # B. Center Crop (Deterministic)
        patch_size = predictor.configuration_manager.patch_size
        img_shape = data.shape[1:]
        patch_coords = tuple([slice((d - p)//2, (d - p)//2 + p) for d, p in zip(img_shape, patch_size)])
        
        full_slice = (slice(None),) + patch_coords
        input_tensor = torch.from_numpy(data[full_slice]).unsqueeze(0).to('cuda', non_blocking=True)
        seg_tensor = torch.from_numpy(seg[full_slice]).unsqueeze(0).to('cuda', non_blocking=True).float()

        # C. Forward Pass with Hook
        feature_hook = FeatureHook(conv_to_hook)
        with torch.no_grad():
            predictor.network(input_tensor)
        
        feature_map = feature_hook.features
        feature_hook.close()

        # D. Downsample GT and Flatten
        seg_down = F.interpolate(seg_tensor, size=feature_map.shape[2:], mode='nearest')
        
        # Flatten: (Batch, D, H, W, C) -> (N, C)
        X_case = feature_map.permute(0, 2, 3, 4, 1).reshape(-1, feature_map.shape[1]).cpu().numpy()
        y_case = seg_down.reshape(-1).cpu().numpy().astype(int)

        # E. Accumulate (with subsampling to save RAM)
        # If we have too many points, take a random 20% chunk from this case
        if X_case.shape[0] > 20000:
            idx = np.random.choice(X_case.shape[0], 20000, replace=False)
            all_X.append(X_case[idx])
            all_y.append(y_case[idx])
        else:
            all_X.append(X_case)
            all_y.append(y_case)

    # 3. Aggregate
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    
    # Global Subsampling (if total exceeds limit)
    if final_X.shape[0] > max_voxels:
        print(f"Dataset too large ({final_X.shape[0]} voxels). Subsampling to {max_voxels}...")
        idx = np.random.choice(final_X.shape[0], max_voxels, replace=False)
        final_X = final_X[idx]
        final_y = final_y[idx]

    print(f"Extraction Complete. Final Dataset: {final_X.shape}")
    return final_X, final_y

# ==========================================
# 3. METRICS & VISUALIZATION
# ==========================================

def compute_metrics(X, y, class_map):
    results = {}
    present_classes = np.unique(y)
    valid_map = {k: v for k, v in class_map.items() if k in present_classes}
    
    if len(present_classes) > 1:
        # 1. Calinski-Harabasz (Fast)
        results['Calinski_Harabasz'] = calinski_harabasz_score(X, y)
        
        # 2. Silhouette Score (Slow - O(N^2))
        # We must subsample if N > 10k, otherwise this hangs forever
        if X.shape[0] > 10000:
            print("Subsampling for Silhouette Score calculation (limit 10k)...")
            idx = np.random.choice(X.shape[0], 10000, replace=False)
            results['Silhouette_Score'] = silhouette_score(X[idx], y[idx])
        else:
            results['Silhouette_Score'] = silhouette_score(X, y)
    else:
        results['Calinski_Harabasz'] = 0.0
        results['Silhouette_Score'] = 0.0

    # 3. Centroids
    centroids = []
    labels = []
    for c in valid_map.keys():
        feats = X[y == c]
        if len(feats) > 0:
            centroids.append(np.mean(feats, axis=0))
            labels.append(valid_map[c])
            
    if len(centroids) > 1:
        centroids = np.array(centroids)
        dist_mat = cdist(centroids, centroids, metric='cosine')
        df_dist = pd.DataFrame(dist_mat, index=labels, columns=labels)
        results['Centroid_Dist_Matrix'] = df_dist
    else:
        results['Centroid_Dist_Matrix'] = None

    return results

def visualize_and_report(X, y, metrics, class_map, title_suffix="stage2"):
    # Subsample strictly for t-SNE
    indices = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if len(idx_c) > 500:
            chosen = np.random.choice(idx_c, 500, replace=False)
        else:
            chosen = idx_c
        indices.extend(chosen)
    
    X_sub = X[indices]
    y_sub = y[indices]
    
    print(f"Running t-SNE on subset of {len(X_sub)} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_emb = tsne.fit_transform(X_sub)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    labels_str = [class_map.get(i, f"Class {i}") for i in y_sub]
    sns.scatterplot(
        x=X_emb[:,0], y=X_emb[:,1], hue=labels_str, 
        palette="tab10", s=15, alpha=0.7, ax=axes[0]
    )
    axes[0].set_title(f"t-SNE Projection {title_suffix}")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    if metrics['Centroid_Dist_Matrix'] is not None:
        sns.heatmap(
            metrics['Centroid_Dist_Matrix'], annot=True, cmap="viridis", 
            fmt=".2f", ax=axes[1]
        )
        axes[1].set_title(f"Inter-Class Centroid Distances {title_suffix}\n(Yellow/Higher is Better)")
    
    plt.tight_layout()
    
    save_path = "/content/drive/MyDrive/PSS Project/nnUNet project/features/stage2.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved to: {save_path}")
    display(Image(filename=save_path))

    print("\n" + "="*40)
    print(f"       AGGREGATED REPORT {title_suffix}       ")
    print("="*40)
    print(f"Global Silhouette Score (Range -1 to 1):      {metrics['Silhouette_Score']:.4f}")
    print(f"Intra-Class Compactness (Calinski-Harabasz):  {metrics['Calinski_Harabasz']:.2f}")
    
    if metrics['Centroid_Dist_Matrix'] is not None:
        print("-" * 40)
        print("Inter-Class Distances (Summary):")
        print(metrics['Centroid_Dist_Matrix'])

# --- USER CONFIGURATION ---
if __name__ == "__main__":

    np.random.seed(42)  
    torch.manual_seed(42)
    
    # PATHS
    #model_folder = '/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_results/Dataset505_Full/nnUNetTrainerCustom1SegUpdate__nnUNetResEncUNetMPlans__3d_fullres'
    model_folder = '/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_results/Dataset505_Full/nnUNetTrainerTripletSinkhornPUpdate__nnUNetResEncUNetMPlans__3d_fullres'
    image_files = [['/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_raw/Dataset505_Full/imagesTs/btcv_case_0007_0000.nii.gz'],['/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_raw/Dataset505_Full/imagesTs/btcv_case_0008_0000.nii.gz'],['/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_raw/Dataset505_Full/imagesTs/btcv_case_0009_0000.nii.gz']] # List format required
    gt_files = ['/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_raw/Dataset505_Full/labelsTs/btcv_case_0007.nii.gz','/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_raw/Dataset505_Full/labelsTs/btcv_case_0008.nii.gz','/content/drive/MyDrive/PSS Project/nnUNet project/nnUNet_raw/Dataset505_Full/labelsTs/btcv_case_0009.nii.gz']
 
    class_map = {
        0: 'Background',
        1: 'Pancreas',
        2: 'Kidneys',
        3: 'Liver',
        4: 'Spleen',
    }
    TARGET_STAGE = 2

    # RUN PIPELINE
    # 1. Extract
    # --- RUN ---
    X, y = extract_dataset_features(
        model_folder, 
        image_files, 
        gt_files, 
        target_stage=TARGET_STAGE
    )
    
    # 2. Compute
    metrics = compute_metrics(X, y, class_map)
    
    # 3. Visualize
    visualize_and_report(X, y, metrics, class_map)
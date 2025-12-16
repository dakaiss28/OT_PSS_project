import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from geomloss import SamplesLoss

torch.manual_seed(0)

def compute_kmeans_centroids(features, K, num_iters=10):
    """
    returns K means centroids from the features using K-means clustering.
    features: (N, D)
    """
    N, D = features.shape
    if N < K:
        return features.mean(dim=0, keepdim=True).repeat(K, 1)

    indices = torch.randperm(N)[:K]
    centroids = features[indices]  # (K, D)

    for _ in range(num_iters):
        dists = torch.cdist(features, centroids, p=2)
        assign = torch.argmin(dists, dim=1)

        new_centroids = []
        for k in range(K):
            cluster_pts = features[assign == k]
            if cluster_pts.numel() == 0:
                new_centroids.append(features[torch.randint(0, N, (1,))])
            else:
                new_centroids.append(cluster_pts.mean(dim=0, keepdim=True))
        centroids = torch.cat(new_centroids, dim=0)

    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids


class LossTriplet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        proto_dim: int = 32,
        device: str = "cuda",
        K_max: int = 5,
        anchor_strength: float = 1.0,
        tau: float = 0.1,
        triplet_margin: float = 1,
        negative_mode: str = "semi_hard",
        lambda_compact: float = 0.1,
        max_anchors: int = 2048,
    ):
        super().__init__()
        assert negative_mode in ["hard", "semi_hard"]
        self.device = torch.device(device)
        self.num_classes = num_classes  
        self.K_max = K_max
        self.proto_dim = proto_dim
        self.anchor_strength = anchor_strength
        self.tau = tau
        self.triplet_margin = triplet_margin
        self.negative_mode = negative_mode
        self.eps = 1e-6
        self.lambda_compact = lambda_compact
        self.max_anchors = max_anchors

        # --- Prototypes initialization ---
        protos = torch.randn(self.num_classes, self.K_max, self.proto_dim)
        protos = F.normalize(protos, p=2, dim=-1)
        self.prototypes = nn.Parameter(protos).cuda()  

    # ----------- Utils -----------
    def _prepare_features(self, feats: torch.Tensor) -> torch.Tensor:
        return feats.permute(0, 2, 3, 4, 1).contiguous().view(-1, feats.shape[1])

    def _prepare_masks(self, masks: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        masks = masks.unsqueeze(1).to(self.device).float()
        masks = F.interpolate(masks, size=target_size, mode="nearest").squeeze(1).long()
        return masks.view(-1)

    # ----------- Forward -----------
    def forward(self, feats, pseudo_masks, gt_masks, labeled_batch):
        B, Df, Zf, Yf, Xf = feats.shape
        target_size = (Zf, Yf, Xf)

        feats_flat = self._prepare_features(feats)
        feats_flat = F.normalize(feats_flat, p=2, dim=-1)
        gt_masks_flat = self._prepare_masks(gt_masks, target_size)
        pseudo_masks_flat = self._prepare_masks(pseudo_masks, target_size)

        protos = F.normalize(self.prototypes, p=2, dim=-1)

        # --- 1. Prototype anchoring with "means" ---
        anchor_loss = torch.tensor(0.0, device=self.device)
        anchor_count = 0

        for c in labeled_batch:
            if c == 0 or c >= self.num_classes:
                continue

            mask_idx = (gt_masks_flat == c).nonzero(as_tuple=False).squeeze(1)
            if mask_idx.numel() == 0:
                continue

            feats_c = feats_flat[mask_idx]  # (Nc, D)
            if feats_c.shape[0] < self.K_max:
                centroid = feats_c.mean(dim=0, keepdim=True)  # (1, D)
                centroids = centroid.repeat(self.K_max, 1)
            else:
                centroids = compute_kmeans_centroids(feats_c, self.K_max, num_iters=5)

            centroids = F.normalize(centroids, p=2, dim=-1)
            class_protos = protos[c]  # (K_max, D)

            loss_c = F.mse_loss(class_protos, centroids)
            anchor_loss += loss_c
            anchor_count += 1

        if anchor_count > 0:
            anchor_loss /= anchor_count
        else:
            anchor_loss = torch.tensor(0.0, device=self.device)

        compact_loss = 0.0
        for c in range(1,self.num_classes):
            proto_class = protos[c]
            if proto_class is None or proto_class.shape[0] < 2:
                continue
            compact_loss += proto_class.var(dim=0).mean()


        # -- 2. Anchor sampling ---
        anchor_idxs = torch.nonzero(pseudo_masks_flat > 0, as_tuple=False).squeeze(1)
        if anchor_idxs.numel() == 0:
            return self.anchor_strength * anchor_loss +  self.lambda_compact * compact_loss

        
        if anchor_idxs.numel() > self.max_anchors:
            perm = torch.randperm(anchor_idxs.numel(), device=self.device)[:self.max_anchors]
            anchor_idxs = anchor_idxs[perm]

        anchor_feats = feats_flat[anchor_idxs]
        anchor_labels = pseudo_masks_flat[anchor_idxs].long()

        anchors_all, pos_all, neg_all = [], [], []
        unique_classes = torch.unique(anchor_labels)

        for c in unique_classes.tolist():
            if c == 0:
                continue
            mask_c = anchor_labels == c
            if not mask_c.any():
                continue

            anchors_c = anchor_feats[mask_c]
            pos_protos = protos[c]
            sim_pos = torch.matmul(anchors_c, pos_protos.t())
            best_k = torch.argmax(sim_pos, dim=1)
            pos_vecs = pos_protos[best_k]

            neg_protos_list = [protos[c2] for c2 in range(self.num_classes) if c2 != c]
            if not neg_protos_list:
                continue
            neg_protos = torch.cat(neg_protos_list, dim=0)

            d_pos = torch.cdist(anchors_c, pos_protos, p=2)
            pos_dists, _ = d_pos.min(dim=1)
            d_neg = torch.cdist(anchors_c, neg_protos, p=2)

            if self.negative_mode == "hard":
                minneg_idx = torch.argmin(d_neg, dim=1)
                neg_vecs = neg_protos[minneg_idx]
            else:  # semi-hard negatives
                neg_vecs = []
                for i in range(d_neg.shape[0]):
                    pos_d = pos_dists[i]
                    mask_valid = (d_neg[i] > pos_d) & (d_neg[i] < pos_d + self.triplet_margin)
                    valid_ids = mask_valid.nonzero(as_tuple=False).flatten()
                    if valid_ids.numel() == 0:
                        j = torch.randint(0, d_neg.shape[1], (1,), device=self.device)
                    else:
                        j = valid_ids[torch.randint(0, valid_ids.numel(), (1,), device=self.device)]
                    neg_vecs.append(neg_protos[j])
                neg_vecs = torch.cat(neg_vecs, dim=0)

            anchors_all.append(anchors_c)
            pos_all.append(pos_vecs)
            neg_all.append(neg_vecs)

        if not anchors_all:
            proto_loss = torch.tensor(0.0, device=self.device)
        else:
            anchors_all = F.normalize(torch.cat(anchors_all, dim=0), p=2, dim=-1)
            pos_all = F.normalize(torch.cat(pos_all, dim=0), p=2, dim=-1)
            neg_all = F.normalize(torch.cat(neg_all, dim=0), p=2, dim=-1)

            d_pos = torch.norm(anchors_all - pos_all, p=2, dim=1)
            d_neg = torch.norm(anchors_all - neg_all, p=2, dim=1)
            proto_loss = F.softplus(d_pos - d_neg + self.triplet_margin).mean()

        total_loss = proto_loss + (self.anchor_strength * anchor_loss) +( self.lambda_compact * compact_loss)
        return total_loss


class LossTripletSinkhorn(LossTriplet):
    def __init__(
        self,
        num_classes: int = 5,
        device: str = "cuda",
        momentum: float = 0.8,
        lambda_thr: float = 1,
        K_max: int = 5,
        triplet_margin: float = 1,
        sinkhorn_blur: float = 0.03,
        sinkhorn_scaling: float = 0.9,
        sinkhorn_mode: str = "sinkhorn",  
        n_samples: int = 2048,
        max_anchors: int = 4096,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.__name__ = "LossTripletSinkhorn"
        self.max_anchors = max_anchors
        self.n_samples = n_samples


        # Sinkhorn divergence loss
        self.sinkhorn = SamplesLoss(
            loss=sinkhorn_mode,
            p=2,
            blur=sinkhorn_blur,
            scaling=sinkhorn_scaling,
            debias=True
        )

    # ---------------- Forward with Sinkhorn Triplet ----------------
    def forward(
        self,
        feats: torch.Tensor,
        pseudo_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        labeled_batch: List[int],
    ):
        B, Df, Zf, Yf, Xf = feats.shape
        target_size = (Zf, Yf, Xf)

        feats_flat = self._prepare_features(feats)
        feats_flat = F.normalize(feats_flat, p=2, dim=-1)
        gt_masks_flat = self._prepare_masks(gt_masks, target_size)
        pseudo_masks_flat = self._prepare_masks(pseudo_masks, target_size)

        protos = F.normalize(self.prototypes, p=2, dim=-1)

        # --- 1. Prototype anchoring with "means" ---
        anchor_loss = torch.tensor(0.0, device=self.device)
        anchor_count = 0

        for c in labeled_batch:
            if c == 0 or c >= self.num_classes:
                continue

            mask_idx = (gt_masks_flat == c).nonzero(as_tuple=False).squeeze(1)
            if mask_idx.numel() == 0:
                continue

            feats_c = feats_flat[mask_idx]  # (Nc, D)
            if feats_c.shape[0] < self.K_max:
                centroid = feats_c.mean(dim=0, keepdim=True)  # (1, D)
                centroids = centroid.repeat(self.K_max, 1)
            else:
                centroids = compute_kmeans_centroids(feats_c, self.K_max, num_iters=5)

            centroids = F.normalize(centroids, p=2, dim=-1)
            class_protos = protos[c]  # (K_max, D)

            loss_c = F.mse_loss(class_protos, centroids)
            anchor_loss += loss_c
            anchor_count += 1

        if anchor_count > 0:
            anchor_loss /= anchor_count
        else:
            anchor_loss = torch.tensor(0.0, device=self.device)

        compact_loss = 0.0
        for c in range(1,self.num_classes):
            proto_class = protos[c]
            if proto_class is None or proto_class.shape[0] < 2:
                continue
            compact_loss += proto_class.var(dim=0).mean()


        # --- 2. Anchor sampling ---
        anchor_idxs = torch.nonzero(pseudo_masks_flat > 0, as_tuple=False).squeeze(1)
        if anchor_idxs.numel() == 0:
            return self.anchor_strength * anchor_loss + self.lambda_compact * compact_loss

        
        if anchor_idxs.numel() > self.max_anchors:
            perm = torch.randperm(anchor_idxs.numel(), device=self.device)[:self.max_anchors]
            anchor_idxs = anchor_idxs[perm]

        anchor_feats = feats_flat[anchor_idxs]
        anchor_labels = pseudo_masks_flat[anchor_idxs].long()

        loss_total = 0.0
        valid_classes = 0

        for c in torch.unique(anchor_labels).tolist():
            if c == 0 or protos[c] is None:
                continue

            mask_c = anchor_labels == c
            anchors_c = anchor_feats[mask_c]

            # --- Positive and Negative sets ---
            pos_protos = protos[c]

            neg_protos_list = [
                protos[c2]
                for c2 in range(self.num_classes)
                if c2 != c and protos[c2] is not None
            ]
            if not neg_protos_list:
                continue
            neg_protos = torch.cat(neg_protos_list, dim=0)

            # Random subset for OT efficiency
            def subsample(X):
                if X.shape[0] > self.n_samples:
                    idx = torch.randperm(X.shape[0], device=self.device)[:self.n_samples]
                    return X[idx]
                return X

            A = subsample(anchors_c)
            P = subsample(pos_protos)
            N = subsample(neg_protos)

            # --- Sinkhorn distances ---
            D_pos = self.sinkhorn(A, P)
            D_neg = self.sinkhorn(A, N)

            #loss_c = F.relu(D_pos - D_neg + self.triplet_margin)
            loss_c = F.softplus(D_pos - D_neg + self.triplet_margin)
            loss_total += loss_c
            valid_classes += 1

        if valid_classes == 0:
            self.anchor_strength * anchor_loss + self.lambda_compact * compact_loss
        proto_loss = loss_total / valid_classes
        return proto_loss + (self.anchor_strength * anchor_loss) + (self.lambda_compact * compact_loss)



class LossSinkhorn(LossTriplet):
    def __init__(
        self,
        num_classes: int = 5,
        device: str = "cuda",
        momentum: float = 0.8,
        lambda_thr: float = 1,
        K_max: int = 5,
        triplet_margin: float = 1,
        sinkhorn_blur: float = 0.03,
        sinkhorn_scaling: float = 0.9,
        sinkhorn_mode: str = "sinkhorn",  
        n_samples: int = 2048,
        max_anchors: int = 4096,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.__name__ = "LossSinkhorn"
        
        self.n_samples = n_samples
        self.max_anchors = max_anchors


        # Sinkhorn divergence loss
        self.sinkhorn = SamplesLoss(
            loss=sinkhorn_mode,
            p=2,
            blur=sinkhorn_blur,
            scaling=sinkhorn_scaling,
            debias=True
        )

    # ---------------- Forward with Sinkhorn  ----------------
    def forward(
        self,
        feats: torch.Tensor,
        pseudo_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        labeled_batch: List[int],
    ):
        B, Df, Zf, Yf, Xf = feats.shape
        target_size = (Zf, Yf, Xf)

        feats_flat = self._prepare_features(feats)
        feats_flat = F.normalize(feats_flat, p=2, dim=-1)
        gt_masks_flat = self._prepare_masks(gt_masks, target_size)
        pseudo_masks_flat = self._prepare_masks(pseudo_masks, target_size)

        protos = F.normalize(self.prototypes, p=2, dim=-1)

        # -- 1. Prototype anchoring with "means" ---
        anchor_loss = torch.tensor(0.0, device=self.device)
        anchor_count = 0

        for c in labeled_batch:
            if c == 0 or c >= self.num_classes:
                continue

            mask_idx = (gt_masks_flat == c).nonzero(as_tuple=False).squeeze(1)
            if mask_idx.numel() == 0:
                continue

            feats_c = feats_flat[mask_idx]  # (Nc, D)
            if feats_c.shape[0] < self.K_max:
                centroid = feats_c.mean(dim=0, keepdim=True)  # (1, D)
                centroids = centroid.repeat(self.K_max, 1)
            else:
                centroids = compute_kmeans_centroids(feats_c, self.K_max, num_iters=5)

            centroids = F.normalize(centroids, p=2, dim=-1)
            class_protos = protos[c]  # (K_max, D)

            loss_c = F.mse_loss(class_protos, centroids)
            anchor_loss += loss_c
            anchor_count += 1

        if anchor_count > 0:
            anchor_loss /= anchor_count
        else:
            anchor_loss = torch.tensor(0.0, device=self.device)

        compact_loss = 0.0
        for c in range(1,self.num_classes):
            proto_class = protos[c]
            if proto_class is None or proto_class.shape[0] < 2:
                continue
            compact_loss += proto_class.var(dim=0).mean()


        # -- 2. Anchor sampling ---
        anchor_idxs = torch.nonzero(pseudo_masks_flat > 0, as_tuple=False).squeeze(1)
        if anchor_idxs.numel() == 0:
            return self.anchor_strength * anchor_loss + self.lambda_compact * compact_loss

        if anchor_idxs.numel() > self.max_anchors:
            perm = torch.randperm(anchor_idxs.numel(), device=self.device)[:self.max_anchors]
            anchor_idxs = anchor_idxs[perm]

        anchor_feats = feats_flat[anchor_idxs]
        anchor_labels = pseudo_masks_flat[anchor_idxs].long()

        loss_total = 0.0
        valid_classes = 0

        for c in torch.unique(anchor_labels).tolist():
            if c == 0 or protos[c] is None:
                continue

            mask_c = anchor_labels == c
            anchors_c = anchor_feats[mask_c]

            # --- Positive set only ---
            pos_protos = protos[c]

           
            def subsample(X):
                if X.shape[0] > self.n_samples:
                    idx = torch.randperm(X.shape[0], device=self.device)[:self.n_samples]
                    return X[idx]
                return X

            A = subsample(anchors_c)
            P = subsample(pos_protos)
          
            loss_c = self.sinkhorn(A, P)
            loss_total += loss_c
            valid_classes += 1

        if valid_classes == 0:
            self.anchor_strength * anchor_loss + self.lambda_compact * compact_loss
        proto_loss = loss_total / valid_classes
        return proto_loss + (self.anchor_strength * anchor_loss) + (self.lambda_compact * compact_loss)

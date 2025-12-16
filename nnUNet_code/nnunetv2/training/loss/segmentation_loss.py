import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import  MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._dice_loss_fn = MemoryEfficientSoftDiceLoss(do_bg = False)
        self._ce_loss_fn = RobustCrossEntropyLoss()

    def _combined_loss(self, logits_sup, target_mask):
        """
        Calculates the combined Dice + CE loss for the supervised part.
        
        Args:
            logits_sup (torch.Tensor): Logits for the supervised classes [B, C_sup, D, H, W].
            target_mask (torch.Tensor): Ground truth mask [B, 1, D, H, W].
        """
        probs_sup = F.softmax(logits_sup, dim=1) 
        dice_loss = self._dice_loss_fn(probs_sup, target_mask)
        ce_loss = self._ce_loss_fn(logits_sup, target_mask[:,0].long())
        return dice_loss + ce_loss
    
    def forward(self, logits, y,labeled_batch):
        probs = F.softmax(logits, dim=1)
        B, K, D, H, W = probs.shape
        total_loss = None

        for b in range(B):
            y_b = y[b]
            probs_b = probs[b]
            logits_b = logits[b]

            labeled = labeled_batch[b]
            labeled = torch.tensor([labeled], device=logits.device)
            unlabeled = torch.tensor([c for c in range(1, K) if c not in labeled],
                                     device=logits.device)
              
            unlabeled_logits = logits_b[unlabeled]
            bg_and_unlabeled_logits = torch.cat([logits_b[0:1], unlabeled_logits], dim=0)
            new_bg_logit = torch.max(bg_and_unlabeled_logits, dim=0, keepdim=True)[0]
            labeled_logits = logits_b[labeled]

            supervised_logits = torch.cat((new_bg_logit, labeled_logits), dim=0).unsqueeze(0)

            # Remap the ground truth labels 
            y_b_remapped = torch.zeros_like(y_b)
            y_b_remapped[y_b == self.bg_index] = 0
            for i, label_val in enumerate(labeled):
                y_b_remapped[y_b == label_val] = i + 1
            y_b_remapped = y_b_remapped.unsqueeze(0).unsqueeze(0)

            sup_loss = self._combined_loss(supervised_logits, y_b_remapped)
           
            batch_loss = sup_loss 
            if total_loss is None:
                total_loss = batch_loss
            else:
                total_loss = total_loss + batch_loss   

        return total_loss / B 
     


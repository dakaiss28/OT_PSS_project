from nnunetv2.training.loss.segmentation_loss import SegmentationLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nan,  autocast
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs
from typing import  List
import numpy as np

class nnUNetTrainerCustom(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.segloss_fn = SegmentationLoss()
        self.feature_hook = None
        self.enable_deep_supervision = False
        self.save_every = 5
        self.initial_lr = 1e-3

    def get_labeled_organs(self,organs_list):
      labels_list = []
      for organ_name in organs_list :
        if organ_name.startswith('pancreas'):
          labels_list.append(1)
        elif organ_name.startswith('kits'):
          labels_list.append(2)
        elif organ_name.startswith('lits'):
          labels_list.append(3)
        elif organ_name.startswith('spleen'):
          labels_list.append(4)
      return labels_list

    def _build_loss(self):
        return self.segloss_fn
    
    def train_step(self, batch: dict, batch_id) -> dict:
        data = batch['data']
        target = batch['target']

        labeled = self.get_labeled_organs(batch['keys'])

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            target = target.squeeze(1).long()
            l = self.compute_loss(output,target,labeled)          

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict, batch_id) -> dict:
        data_batch = batch['data']
        target_batch = batch['target']

        labeled_batch = self.get_labeled_organs(batch['keys'])
        dice_batch = []
        loss = 0

        for b in range(data_batch.shape[0]):
          data = data_batch[b:b+1]
          target = target_batch[b:b+1]
          labeled = labeled_batch[b:b+1]


          #print("Dataset for this sample :", labeled)
          data = data.to(self.device, non_blocking=True)
          if isinstance(target, list):
              target = [i.to(self.device, non_blocking=True) for i in target]
          else:
              target = target.to(self.device, non_blocking=True)

          # Autocast can be annoying
          # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
          # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
          # So autocast will only be active if we have a cuda device.
          with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
              output = self.network(data)
              del data
              target = target.squeeze(1).long()
              l = self.compute_loss(output,target,labeled)

          # we only need the output with the highest output resolution (if DS enabled)
          if self.enable_deep_supervision:
              output = output[0]
              target = target[0]

          # the following is needed for online evaluation. Fake dice (green line)
          axes = [0] + list(range(2, output.ndim))

          if self.label_manager.has_regions:
              predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
          else:
              # no need for softmax
              output_seg = output.argmax(1)[:, None]
              predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
              predicted_segmentation_onehot.scatter_(1, output_seg, 1)
              del output_seg

          if self.label_manager.has_ignore_label:
              if not self.label_manager.has_regions:
                  mask = (target != self.label_manager.ignore_label).float()
                  # CAREFUL that you don't rely on target after this line!
                  target[target == self.label_manager.ignore_label] = 0
              else:
                  if target.dtype == torch.bool:
                      mask = ~target[:, -1:]
                  else:
                      mask = 1 - target[:, -1:]
                  # CAREFUL that you don't rely on target after this line!
                  target = target[:, :-1]
          else:
              mask = None

          tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

          tp_hard = tp.detach().cpu().numpy()
          fp_hard = fp.detach().cpu().numpy()
          fn_hard = fn.detach().cpu().numpy()
          if not self.label_manager.has_regions:
              # if we train with regions all segmentation heads predict some kind of foreground. In conventional
              # (softmax training) there needs tobe one output for the background. We are not interested in the
              # background Dice
              # [1:] in order to remove background
              tp_hard = tp_hard[1:]
              fp_hard = fp_hard[1:]
              fn_hard = fn_hard[1:]

          dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_hard, fp_hard, fn_hard)]]

          
          for c in [1,2,3,4]:
            if c not in labeled :
              dc_per_class[c-1] = np.nan
          
          dice_batch.append(np.array(dc_per_class, dtype=np.float32))

          loss += l.detach().cpu().numpy()
        
        #print(dice_batch)
        dice_batch = np.nanmean(dice_batch, axis=0)
        #print("mean dice batch : ", dice_batch)
        loss /= data_batch.shape[0]

        return {'loss': loss, 'dice' : dice_batch }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        global_dc_per_class = np.nanmean(outputs_collated['dice'], 0)

        loss_here = np.mean(outputs_collated['loss'])

        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    
    def compute_loss(self, data, target,labeled):
        logits = data
        loss = self.segloss_fn(logits, target,labeled)
        return loss

    def run_training(self):
        self.on_train_start()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                out = self.train_step(next(self.dataloader_train),batch_id)
                if out['loss'] is not None:
                    train_outputs.append(out)
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    out = self.validation_step(next(self.dataloader_val),batch_id)
                    if out['loss'] is not None:
                        val_outputs.append(out)
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()


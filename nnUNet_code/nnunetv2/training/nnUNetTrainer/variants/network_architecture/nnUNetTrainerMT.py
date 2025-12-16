from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerOTLayer import nnUNetTrainerOTLayer
from nnunetv2.training.loss.ot_loss import SoftOTLossUnc
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from typing import Tuple, Union, List
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from typing import List
from torch import nan, nn, autocast
import torch.nn.functional as F
import copy 
import numpy as np
import pydoc

class SelectiveDropoutUNet(ResidualEncoderUNet):
    """
    This class inherits from nnU-Net's default ResidualEncoderUNet and adds
    dropout layers specifically to the last two stages of the encoder.
    """
    def __init__(self, *args, **kwargs):
        # First, initialize the parent class with all its original arguments
        super().__init__(*args, **kwargs)
        

        # Define our dropout layers. We'll use 3D dropout.
        # You can adjust the dropout probability 'p' as needed.
        dropout_prob = 0.2
        self.dropout_stage_minus_1 = nn.Dropout3d(p=dropout_prob)
        self.dropout_stage_bottleneck = nn.Dropout3d(p=dropout_prob)

        print(f"Custom Network Initialized: Added Dropout3d(p={dropout_prob}) to the last two encoder stages.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        We override the forward pass to intercept the skip connections from the encoder
        and apply our custom dropout layers before passing them to the decoder.
        """
        # 1. Get the skip connections from the original encoder
        skips = self.encoder(x)

        # 2. Apply dropout to the features from the last two stages
        # The last item in the 'skips' list is the bottleneck
        skips[-1] = self.dropout_stage_bottleneck(skips[-1])
        # The second to last item is the one before the bottleneck
        skips[-2] = self.dropout_stage_minus_1(skips[-2])

        # 3. Pass the modified skip connections to the decoder
        return self.decoder(skips)

class nnUNetTrainerMT(nnUNetTrainerOTLayer):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.otloss_fn = SoftOTLossUnc(device=self.device)
        self.teacher_network = copy.deepcopy(self.network)
        self.ema_decay = 0.999
        
    def initialize(self):
      super().initialize()
      self.teacher_network = copy.deepcopy(self.network)
      for param in self.teacher_network.parameters():
            param.requires_grad = False
      self.print_to_log_file(f"Initialized Mean Teacher with EMA decay: {self.ema_decay}")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
                
        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision

        network = SelectiveDropoutUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network
        
    @torch.no_grad()
    def _update_teacher_network(self):
        """
        Performs the Exponential Moving Average update for the teacher network.
        Formula: teacher_param = decay * teacher_param + (1 - decay) * student_param
        """
        student_params = self.network.parameters()
        teacher_params = self.teacher_network.parameters()

        for student_p, teacher_p in zip(student_params, teacher_params):
            teacher_p.data.mul_(self.ema_decay).add_(student_p.data, alpha=1 - self.ema_decay)

    @torch.no_grad()
    def get_uncertaintyMC(self, data: torch.Tensor, n_passes: int = 5):
        """
        Performs prediction using MC Dropout on the TEACHER model.

        Args:
            data (torch.Tensor): The input tensor (e.g., a batch of images).
            n_passes (int): The number of stochastic forward passes for MC Dropout.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            - The mean prediction (softmax probabilities).
            - The uncertainty map (pixel-wise entropy).
        """
        self.teacher_network.train()  # Activate dropout layers for stochasticity
        
        predictions = []
        for _ in range(n_passes):
            # We use autocast for potential performance gains with mixed precision
            with autocast(self.device.type, enabled=True):
                logits = self.teacher_network(data)
                # Convert logits to probabilities and store them
                predictions.append(F.softmax(logits, dim=1))
        
        # Stack predictions along a new dimension and calculate the mean
        # Shape: (n_passes, B, C, H, W, D) -> (B, C, H, W, D)
        mean_prediction = torch.stack(predictions).mean(dim=0)
        
        # Calculate pixel-wise entropy on the mean prediction
        uncertainty_map = self.calculate_entropy(mean_prediction)

        self.teacher_network.eval() # Set teacher back to eval mode
        
        return mean_prediction, uncertainty_map

    @staticmethod
    def calculate_entropy(probabilities: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Calculates the Shannon entropy for each pixel in a probability map.

        Args:
            probabilities (torch.Tensor): A tensor of softmax probabilities (B, C, H, W, D).
            epsilon (float): A small value to prevent log(0).

        Returns:
            torch.Tensor: An entropy map (B, 1, H, W, D).
        """
        # H(p) = - sum(p * log2(p)) over classes
        entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=1, keepdim=True)
        return entropy
    
    def compute_loss(self, data, target,labeled,uncertainty):
        logits = data
        feats = self.feature_hook.features  # bottleneck features

        loss = self.segloss_fn(logits, target,labeled) + self.alpha * self.otloss_fn(feats, logits, target,labeled,uncertainty)
        return loss 

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        labeled = self.get_labeled_organs(batch['keys'])  # e.g., [1, 2]

        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            target = target.squeeze(1).long()

            # segmentation loss always applied
            seg_loss = self.segloss_fn(output, target, labeled)

            feats = self.feature_hook.features
            _, uncertainty = self.get_uncertaintyMC(data)
            ot_loss = self.otloss_fn(feats, output, target, labeled, uncertainty)

            loss = seg_loss + self.alpha * ot_loss


        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        self._update_teacher_network()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
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
              _, uncertainty = self.get_uncertaintyMC(data)
              del data
              target = target.squeeze(1).long()
              l = self.compute_loss(output,target,labeled,uncertainty)

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
          
          for c in [1,2,3]:
            if c not in labeled :
              dc_per_class[c-1] = np.nan
          
          dice_batch.append(np.array(dc_per_class, dtype=np.float32))

          loss += l.detach().cpu().numpy()
        
        #print(dice_batch)
        dice_batch = np.nanmean(dice_batch, axis=0)
        #print("mean dice batch : ", dice_batch)
        loss /= data_batch.shape[0]

        return {'loss': loss, 'dice' : dice_batch }



    
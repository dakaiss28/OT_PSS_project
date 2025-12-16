from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCustom import nnUNetTrainerCustom 
from nnunetv2.training.loss.otTriplet_losses import LossTriplet, LossTripletSinkhorn, LossSinkhorn
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader, nnUNetDataLoaderWithPseudoLabels
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetWithPseudoLabels, infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from torch import nan, nn, autocast
from nnunetv2.utilities.helpers import dummy_context


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, model, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


class nnUNetTrainerTriplet(nnUNetTrainerCustom):
  def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        bottleneck_dim = 128
        intermediate_dim = 64
        self.projection_dim = 32 

        self.projector = nn.Sequential(
            nn.Conv3d(bottleneck_dim, intermediate_dim, kernel_size=1),
            nn.BatchNorm3d(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(intermediate_dim, self.projection_dim, kernel_size=1),
        ).to(self.device)

        self.triplet_fn = LossTriplet(
            num_classes=5,
            proto_dim=self.projection_dim,
            K_max=5,
            anchor_strength=1.0,
            device=self.device,
            negative_mode="semi_hard",
            tau=0.1,
            triplet_margin=1,
        )
        self.alpha_triplet = 1
        self.alpha_seg = 1
        self.num_epochs = 100
        self.initial_lr = 1e-4
        self.stage = 2
        self.preprocessed_pseudo_dataset_folder = "path/to/preprocessed/pseudo/dataset/folder"

        self.print_to_log_file("Projector and LossTriplet initialized.")

  def initialize(self):
            super().initialize()
            layer = self.network.decoder.stages[self.stage] # Stage 2 : Dim 128, Stage 3 : Dim 64
            conv_to_hook = getattr(layer, "conv", layer)
            self.feature_hook = FeatureHook(conv_to_hook)

  def configure_optimizers(self):
      optimizer, lr_scheduler = super().configure_optimizers()
      optimizer.add_param_group({'params': self.projector.parameters(), 'lr': 1e-3})
      optimizer.add_param_group({'params': self.triplet_fn.parameters(), 'lr': 1e-3})
      self.print_to_log_file("Added projector and triplet_fn parameters to the optimizer.")
      return optimizer, lr_scheduler

  def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDatasetWithPseudoLabels(
            original_preprocessed_folder=self.preprocessed_dataset_folder,
            pseudo_preprocessed_folder=self.preprocessed_pseudo_dataset_folder,
            identifiers=tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        
        # The validation set does not need pseudo-labels, so we create it the standard way using self.dataset_class.
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder, 
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        return dataset_tr, dataset_val


  def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderWithPseudoLabels(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

  def on_train_epoch_start(self):
      super().on_train_epoch_start()
      self.projector.train()

  def on_validation_epoch_start(self):
      super().on_validation_epoch_start()
      self.projector.eval()


  def train_step(self, batch: dict, batch_id) -> dict:
    data = batch['data']
    target = batch['target']
    pseudo = batch['pseudo_target']
    labeled = self.get_labeled_organs(batch['keys'])  # e.g., [1, 2]

    data = data.to(self.device, non_blocking=True)
    target = target.to(self.device, non_blocking=True)

    self.optimizer.zero_grad(set_to_none=True)

    with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        output = self.network(data)
        target = target.squeeze(1).long()
        pseudo = pseudo.squeeze(1).long()
    
        feats = self.feature_hook.features 
        projected_feats = self.projector(feats)

        seg_loss = self.segloss_fn(output, target, labeled)
        triplet_loss = self.triplet_fn(projected_feats, pseudo, target, labeled)
        loss = (self.alpha_seg * seg_loss) + (self.alpha_triplet * triplet_loss)


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
    

    return {"loss": loss.detach().cpu().numpy()}
  
class nnUNetTrainerTripletSinkhorn(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.alpha_triplet = 1
        self.triplet_fn = LossTripletSinkhorn(device=self.device)


class nnUNetTrainerSinkhorn(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.alpha_triplet = 1
        self.triplet_fn = LossSinkhorn(device=self.device)



# ---------------------------- Ablations -------------------------------------



#------------------------------ K prototypes --------------------------------
class nnUNetTrainerTripletSinkhornPK1(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 1,
        triplet_margin = 1,
        sinkhorn_blur = 0.03,
    )

class nnUNetTrainerTripletSinkhornPK10(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 10,
        triplet_margin = 1,
        sinkhorn_blur = 0.03,
    )

#------------------------------ M margin --------------------------------
class nnUNetTrainerTripletSinkhornPM01(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 5,
        triplet_margin = 0.1,
        sinkhorn_blur = 0.03,
    )


class nnUNetTrainerTripletSinkhornPM15(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 5,
        triplet_margin = 1.5,
        sinkhorn_blur = 0.03,
    )

class nnUNetTrainerTripletSinkhornPM05(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 5,
        triplet_margin = 0.5,
        sinkhorn_blur = 0.03,
    )

class nnUNetTrainerTripletSinkhornPM12(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 5,
        triplet_margin = 1.2,
        sinkhorn_blur = 0.03,
    )


#------------------------------ e blur  --------------------------------
class nnUNetTrainerTripletSinkhornPE1(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 5,
        triplet_margin = 1,
        sinkhorn_blur = 1,
    )


class nnUNetTrainerTripletSinkhornPE01(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 5,
        triplet_margin = 1,
        sinkhorn_blur = 0.1,
    )


class nnUNetTrainerTripletSinkhornPE10(nnUNetTrainerTriplet):
    def __init__(self, plans, configuration, fold, dataset_json, device="cuda"):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.triplet_fn = LossTripletSinkhorn(
        K_max = 5,
        triplet_margin = 1,
        sinkhorn_blur = 10,
    )





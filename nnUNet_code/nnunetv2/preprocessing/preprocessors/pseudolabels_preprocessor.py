import numpy as np
import os
import shutil
from multiprocessing import Pool
from typing import Union, List
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, join, maybe_mkdir_p

# Import the necessary nnU-Net classes
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero


class PseudoLabelPreprocessor(DefaultPreprocessor):
    def run_case_pseudo_label(self,
                              identifier: str,
                              original_preprocessed_dir: str,
                              raw_pseudo_label_dir: str,
                              output_dir: str,
                              plans_manager: PlansManager,
                              configuration_name: str,
                              dataset_json: dict):
        """
        Processes a single case by reusing the preprocessed image and applying transformations
        only to the new pseudo-label.
        """
        # 1. Load the already preprocessed image and its properties
        # We can use the Blosc2 dataset loader for this
        try:
            data, _, _, properties = nnUNetDatasetBlosc2(
                original_preprocessed_dir, [identifier]
            ).load_case(identifier)
            data = np.array(data) # Convert from blosc2 array to numpy
        except FileNotFoundError:
            print(f"  -> Warning: Original preprocessed case {identifier} not found. Skipping.")
            return

        # 2. Load the new pseudo-label from its raw (.nii.gz) file
        rw = plans_manager.image_reader_writer_class()
        pseudo_seg_file = join(raw_pseudo_label_dir, f"{identifier}.nii.gz")
        if not os.path.exists(pseudo_seg_file):
            print(f"  -> Warning: Raw pseudo-label for {identifier} not found. Creating empty seg.")
            # Create an empty segmentation if it's missing, to avoid crashes
            pseudo_seg = np.zeros_like(data, dtype=np.int16)
        else:
            pseudo_seg, _ = rw.read_seg(pseudo_seg_file)

        # 3. Apply the EXACT SAME transformations to the pseudo-label
        # We use the information stored in the properties of the original preprocessed case
        configuration_manager = plans_manager.get_configuration(configuration_name)
        
        # a) Transpose
        pseudo_seg = pseudo_seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
        
        # b) Crop using the SAME bounding box
        bbox = properties['bbox_used_for_cropping']
        pseudo_seg = pseudo_seg[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        
        # c) Resample to the SAME shape
        target_spacing = configuration_manager.spacing
        if len(target_spacing) < len(pseudo_seg.shape[1:]):
            target_spacing = [original_spacing[0]] + list(target_spacing)
        new_shape = compute_new_shape(pseudo_seg.shape[1:], original_spacing, target_spacing)
        pseudo_seg = configuration_manager.resampling_fn_seg(pseudo_seg, new_shape, original_spacing, target_spacing)

        # Ensure the final shape matches the preprocessed image data
        if pseudo_seg.shape[1:] != data.shape[1:]:
            print(f"  -> ERROR: Shape mismatch for {identifier} after preprocessing pseudo-label!")
            print(f"     Data shape: {data.shape}, Pseudo-label shape: {pseudo_seg.shape}")
            return
            
        # Final type conversion
        if np.max(pseudo_seg) > 127:
            pseudo_seg = pseudo_seg.astype(np.int16)
        else:
            pseudo_seg = pseudo_seg.astype(np.int8)

        # 4. Save the original preprocessed image and the NEWLY preprocessed pseudo-label together
        output_filename_truncated = join(output_dir, identifier)
        nnUNetDatasetBlosc2.save_case(
            data=data,
            seg=pseudo_seg,
            properties=properties,
            output_filename_truncated=output_filename_truncated
        )

    def run_repackaging(self,
                        dataset_name_or_id: Union[int, str],
                        configuration_name: str,
                        plans_identifier: str,
                        raw_pseudo_label_dir: str,
                        num_processes: int):
        """
        Main entry point for the repackaging process.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        
        #original_preprocessed_dir = join(nnUNet_preprocessed, dataset_name, configuration_name)
        original_preprocessed_dir = join(nnUNet_preprocessed, dataset_name, f"nnUNetPlans_{configuration_name}")
        output_dir = join(nnUNet_preprocessed, f"{dataset_name}_pseudo", f"nnUNetPlans_{configuration_name}")

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        maybe_mkdir_p(output_dir)

        plans_file = join(nnUNet_preprocessed, dataset_name, f"{plans_identifier}.json")
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        
        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        identifiers = nnUNetDatasetBlosc2.get_identifiers(original_preprocessed_dir)

        # Run in parallel
        pool = Pool(num_processes)
        args_list = [
            (identifier, original_preprocessed_dir, raw_pseudo_label_dir, output_dir, 
             plans_manager, configuration_name, dataset_json)
            for identifier in identifiers
        ]
        
        with tqdm(total=len(identifiers), desc="Repackaging with pseudo-labels") as pbar:
            for _ in pool.starmap(self.run_case_pseudo_label, args_list):
                pbar.update()
        pool.close()
        pool.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Repackage a preprocessed nnU-Net dataset with new pseudo-labels.")
    parser.add_argument("-d", type=str, required=True, help="Dataset name or ID.")
    parser.add_argument("-p", type=str, default='nnUNetResEncUNetMPlans', help="Plans identifier, e.g., nnUNetPlans.")
    parser.add_argument("-c", type=str, default='3d_fullres', help="Configuration name, e.g., 3d_fullres.")
    parser.add_argument("-i_pseudo", type=str, required=True, help="Path to the folder with the new, filtered pseudo-labels in .nii.gz format.")
    parser.add_argument("-j", type=int, default=8, help="Number of parallel processes.")
    args = parser.parse_args()
    
    preprocessor = PseudoLabelPreprocessor()
    preprocessor.run_repackaging(
        dataset_name_or_id=args.d,
        configuration_name=args.c,
        plans_identifier=args.p,
        raw_pseudo_label_dir=args.i_pseudo,
        num_processes=args.j
    )
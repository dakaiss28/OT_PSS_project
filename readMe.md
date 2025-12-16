# Optimal Transport Metric Learning for Feature Alignment in Partially Supervised Segmentation

This repository contains the code related to the paper **"Optimal Transport Metric Learning for Feature Alignment in Partially Supervised Segmentation"**, accepted at the AAAI Workshop on Health Intelligence (W3PHIAI-26).

The code is based on the [nnU-Net repository](https://github.com/MIC-DKFZ/nnUNet).

## Datasets
The datasets used for our experiments are:
- **KiTS19**: [link](https://kits19.grand-challenge.org/)
- **LiTS19**: [link](https://competitions.codalab.org/competitions/17094)
- **MSD Spleen**: [link](http://medicaldecathlon.com/)
- **MSD Pancreas**: [link](http://medicaldecathlon.com/)

Please follow the guidelines available on the [nnU-Net repository](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) to prepare the dataset.

We labeled classes as follows:

| Class Name | Label ID |
| :--- | :--- |
| Background | 0 |
| Pancreas | 1 |
| Kidneys | 2 |
| Liver | 3 |
| Spleen | 4 |

The `dataset.json` as well as the splits used are available [here](./dataset/).

## Training

Again, we refer you to the nnUNet documentation to prepare training [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).

### Stage 1
The files referring to Stage 1 training are:
- [segmentation_loss.py](nnUNet_code/nnunetv2/training/loss/segmentation_loss.py)
- [nnUNetTrainerCustom.py](nnUNet_code/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCustom.py)

To train Stage 1, run the command:
```bash
nnUNet_train 3d_fullres nnUNetTrainerCustom <TASK_ID> <FOLD>
```

After stage 1 is completed, you need to generate the pseudo-labels (see Prediction section). After that, form a new dataset with the scans and pseudo-labels, and apply preprocessing.  

### Stage 2

For Stage 2, training starts with the best checkpoint from Stage 1.

The OT Triplet losses are available here: [otTriplet_losses.py](nnUNet_code/nnunetv2/training/loss/otTriplet_losses.py), and the trainer associated is [nnUNetTrainerOTTriplet.py](nnUNet_code/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerOTTriplet.py).

/!\ You need to update the path to the pseudo labels dataset in nnUNetTrainerOTTriplet.py

To train Stage 2, run the command:

```bash
nnUNet_train 3d_fullres nnUNetTrainerTripletSinkhorn <TASK_ID> <FOLD> \
-pretrained_weights <PATH_TO_STAGE1_CHECKPOINT>

```

## Prediction

Run the following command to run inference on test cases:

```bash
nnUNet_predict -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -t <TASK_ID> -m 3d_fullres -tr nnUNetTrainerTripletSinkhorn
```
To evaluate the test cases, run : 

```bash
nnUNetv2_evaluate_folder  <GROUD_TRUTH_FOLDER> <PREDICTION_FOLDER> -djfile <DATASET_FILE_IN_PREDICTION_FOLDER> -pfile <PLANS_FILE_IN_PREDICTION_FOLDER>
```


## Citation 

If you use this code, please cite :

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```
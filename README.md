# LIDChallenge2020-NoPeopleAllowed
A 3rd place solution for [LID Challenge at CVPR 2020](https://lidchallenge.github.io/) on Weakly Supervised Semantic Segmentation. 

An implementation of the [NoPeopleAllowed: The Three-Step Approach to Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2006.07601) by Mariia Dobko, Ostap Viniavskiy, and Oles Dobosevych.

<p align="center"><img src="./images/examples.png" alt="outline" width="90%"></p>

## Overview
We propose an approach to weakly supervised semantic segmentation, which consists of three consecutive steps.
The first two steps extract high-quality pseudo masks from image-level annotated data, which are then used to train a
segmentation model on the third step.

The presented approach also addresses two problems in the data: class imbalance and missing labels. Using only image-level
 annotations as supervision, our method is capable of segmenting various classes and complex objects. It achieves 37.34 mean 
 IoU on the test set, placing 3rd at the LID Challenge in the task of weakly supervised semantic segmentation.


## Data
Data is provided by a challenge organizers. The training dataset is available at [Imagenet DET](http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET.tar.gz), 
val and test dataset are available at [Google Drive](https://drive.google.com/open?id=1B0enLzxyIULbRZWUi0XpNnXCu7nwFI-f).


## Usage

Step 1. **Classification model training.**  
The first step is to train a classification model using weak labels. The model will be used for extraction of pseudo-segmentation labels.
To train the classification model run the following command:
```
python model_training/cam_generation/train_imagenet.py
```
The path to corresponding config: *model_training/cam_generation/config/imagenet.yaml*  

Step 2. **Multiscale CAM extraction.**  
The next step is to extract Class Activation Maps from the classification model. Extraction can be done using two different techniques: GRAD-CAM and GRAD-CAM++.
Also the given code support extracting CAMs for both predefined class labels, as well as making class predictions on the fly.
To extract CAMs from the classification model run the following command:
```
python inference/irn_steps/multiscale_cam_imagenet.py
```
The path to corresponding config: *inference/irn_steps/config/multiscale_imagenet_cam.yaml*  

Step 3. **CAM refinement using CRF.**  
In this step we use Conditional Random Field to refine the CAMs produced in the previous step. The output of this step will be used to train IRNet model.
To refine CAMs and obtain training data for IRNet run the following command:
```
python inference/irn_steps/crf_cam_processing_imagenet.py
```
The path to corresponding config: *inference/irn_steps/config/crf_cam_processing_imagenet.yaml* 

Step 4. **IRNet training.**  
Next, we train the IRNet model, which will be used for extracting Class Boundary Maps.
To train IRNet model run the following command:
```
python model_training/cam_generation/irn_imagenet.py
```
The path to corresponding config: *model_training/cam_generation/config/irnet_imagenet.yaml* 

Step 5. **IRNet inference.**  
Here, we use the IRNet model to extract Class Boundary Maps. Class Boundary Maps are used to refine the output of Step 2.
To run the inference on IRNet model run the following command:
```
python inference/irn_steps/irn_inference.py
```
The path to corresponding config: *inference/irn_steps/config/irn_inference.yaml* 

Step 6. **Segmentation model training.**  
In this step we train segmentation model using the pseudo-segmentation maps obtained in the previous step.
To train the classification model run the following command:
```
python model_training/cam_generation/train_imagenet.py
```
The path to corresponding config: *model_training/segmentation/config/imagenet.yaml* 

Step 7. **Segmentation model inference.**  
In the final step we run end-to-end inference using segmentation model trained on the previous step.
To make the inference on segmentation model run the following command:
```
python inference/segmentation/TODO.py
```
The path to corresponding config: *inference/segmentation/config/TODO.yaml* 


## Citation
```
@article{dobko2020lid,
     author = {Dobko, Mariia and Viniavskyi, Ostap and Dobosevych, Oles},
     title = {NoPeopleAllowed: The Three-Step Approach to Weakly Supervised Semantic Segmentation},
     journal = {The 2020 Learning from Imperfect Data (LID) Challenge - CVPR Workshops},
     year = {2020}
}
```

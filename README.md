<div align="center">

<h1>Physically consistent and uncertainty-aware learning of spatiotemporal dynamics</h1>

<div>
    <strong>This repository contains the official implementation and pre-trained models of the work "Physically consistent and uncertainty-aware learning of spatiotemporal dynamics".</strong>
</div>

</div>

<div align="center">
<img src="https://github.com/HydroPML/PCNO/blob/main/figures/fig01.png" width="100%"/>
Schematic illustration of the proposed PCNO and DiffPCNO frameworks.

</div>

# ​📥​ Download
## Training data
- The datasets for the KSE, Kolmogorov flow, and atmospheric modeling are available via the [Zenodo repository](https://doi.org/10.5281/zenodo.17410273).  
- The datasets for the flood forecasting are available at the [Zenodo repository](https://doi.org/10.5281/zenodo.14017092).
## Model weights
You can download the model weights for PCNO, DiffPCNO, and PCNO-Refiner from [Link](https://drive.google.com/drive/folders/1EaczWpBe6HK5dDQXSxTSkW8yzl6mO4wE?usp=sharing).
# 🎰 Zero-shot Inference and Sampling
Download the checkpoints and prepare the data.
## 1D Kuramoto–Sivashinsky dynamics
### PCNO
Change the path in `PCNO/PCNO/1D_KSE/experiments_test_PCNO.py` at Line 55: `results_path` to save the results.  
Change the path in `PCNO/PCNO/1D_KSE/experiments_test_PCNO.py` at Line 160: `root =...` to your directory where checkpoints are stored.  
Change the data path in `PCNO/PCNO/1D_KSE/experiments_test_PCNO.py` (lines 234-239) to your directory containing the training, validation, and testing datasets.  
Run `python PCNO/PCNO/1D_KSE/experiments_test_PCNO.py` to predict Kuramoto-Sivashinsky dynamics using PCNO.  
### DiffPCNO


## 2D Kolmogorov turbulent flow

## 2D Real-world flood forecasting

## 2D abd 3D Atmospheric modeling

# 👻​ Training 
## 1D Kuramoto–Sivashinsky dynamics


## 2D Kolmogorov turbulent flow

## 2D Real-world flood forecasting

## 2D abd 3D Atmospheric modeling

# ☄️​ Fine-tuning

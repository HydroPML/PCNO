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
Change the path in `PCNO/PCNO/1D_KSE/experiments_test_PCNO.py` at Line 160: `root=...` to your directory where checkpoints of PCNO are stored.  
Change the data path in `PCNO/PCNO/1D_KSE/experiments_test_PCNO.py` (Lines 234-239) to your directory containing the testing datasets.  
Run `python PCNO/PCNO/1D_KSE/experiments_test_PCNO.py` to predict Kuramoto-Sivashinsky dynamics using PCNO.  
### DiffPCNO
Change the path in `PCNO/DiffPCNO/1D_KSE/KSE_Sampling.py` at Line 57: `results_path` to save the results.  
Change the path in `PCNO/DiffPCNO/1D_KSE/KSE_Sampling.py` at Line 193: `root=...` to your directory where checkpoints of DiffPCNO are stored.  
Change the path in `PCNO/DiffPCNO/1D_KSE/KSE_Sampling.py` at Line 219: `path_model_con=...` to your directory where checkpoints of PCNO are stored.  
Change the data path in `PCNO/DiffPCNO/1D_KSE/KSE_Sampling.py` (Lines 299-306) to your directory containing the testing datasets.  
Change the path in `PCNO/DiffPCNO/1D_KSE/KSE_Sampling.py` at Line 765: `movie_dir=...` to save the visualization results.  
Run `PCNO/DiffPCNO/1D_KSE/KSE_Sampling.py` to obtain prediction results `pred` and uncertainty results `pred_std` using DiffPCNO.  
### PCNO-Refiner
Change the path in `PCNO/PCNO-Refiner/KSE_Sampling.py` at Line 54: `results_path` to save the results.  
Change the path in `PCNO/PCNO-Refiner/KSE_Sampling.py` at Line 189: `root=...` to your directory where checkpoints of PCNO-Refiner are stored.  
Change the path in `PCNO/PCNO-Refiner/KSE_Sampling.py` at Line 214: `path_model_con= ...` to your directory where checkpoints of PCNO are stored.  
Change the data path in `PCNO/PCNO-Refiner/KSE_Sampling.py` (Lines 289-296) to your directory containing the testing datasets.  
Change the path in `PCNO/PCNO-Refiner/KSE_Sampling.py` at Line 708: `movie_dir=...` to save the visualization results.  
Run `PCNO/PCNO-Refiner/KSE_Sampling.py` to obtain prediction results `pred` and uncertainty results `pred_std` using PCNO-Refiner.  

## 2D Kolmogorov turbulent flow
### PCNO
Change the path in `PCNO/PCNO/2D_Kolmogorov/experiments_test_PCNO.py` at Line 54: `results_path` to save the results.    
Change the data path in `PCNO/PCNO/2D_Kolmogorov/experiments_test_PCNO.py` at Line 57: `data_path` to your directory containing the NS datasets.   
Change the path in `PCNO/PCNO/2D_Kolmogorov/experiments_test_PCNO.py` at Line 193: `root=...` to your directory where checkpoints of PCNO are stored.     
Run `PCNO/PCNO/2D_Kolmogorov/experiments_test_PCNO.py` to predict Kolmogorov turbulent flow  in velocity form using PCNO.    
### DiffPCNO
Change the path in `PCNO/DiffPCNO/2D_Kolmogorov/Kolmogorov_sampling.py` at Line 56: `results_path` to save the results.    
Change the data path in `PCNO/DiffPCNO/2D_Kolmogorov/Kolmogorov_sampling.py` at Line 59: `data_path` to your directory containing the NS datasets.   
Change the path in `PCNO/DiffPCNO/2D_Kolmogorov/Kolmogorov_sampling.py` at Line 209: `root=...` to your directory where checkpoints of DiffPCNO are stored.    
Change the path in `PCNO/DiffPCNO/2D_Kolmogorov/Kolmogorov_sampling.py` at Line 237: `path_model_con=...` to your directory where checkpoints of PCNO are stored.    
Change the path in `PCNO/DiffPCNO/2D_Kolmogorov/Kolmogorov_sampling.py` at Line 807 and Line 842: `movie_dir=...` to save the visualization results.    
Run `PCNO/DiffPCNO/2D_Kolmogorov/Kolmogorov_sampling.py` to obtain prediction results `pred` and uncertainty results `pred_std` using DiffPCNO.    
## 2D Real-world flood forecasting
The surrogate models (PCNO and DiffPCNO) are designed for large-scale, cross-regional, and downscaled flood forecasting.  We use FloodCastBench to evaluate surrogate models. The dataset comprises four large-scale
floods: Pakistan flood, Mozambique flood, Australia flood, and UK flood. To assess the effectiveness and transferability of these models, we define two scenarios: low-fidelity forecasting
using the Pakistan and Mozambique flood datasets (480 m spatial, 5 min temporal resolution) and high-fidelity forecasting using the Australia and UK flood datasets (60 m or 30 m spatial, 5 min temporal resolution).  
![FloodCastBench](https://github.com/HydroPML/PCNO/blob/main/figures/fig6.png)
**Welcome to test more flood forecasting scenarios.**
### PCNO
Change the path in `PCNO/PCNO/2D_Flood/experiments_Flood_test.py` at Line 55: `results_path` to save the results.    
Change timesteps `T` (Line 62) and sample sizes (Line 63-65) in `PCNO/PCNO/2D_Flood/experiments_Flood_test.py` to your forecasting scenario.   
Change the space sizes `Sy` and `Sx` (Line 113-127) in `PCNO/PCNO/2D_Flood/experiments_Flood_test.py` according to the flood scenario you want to predict.  
Change the path in `PCNO/PCNO/2D_Flood/experiments_Flood_test.py` at Line 183: `root=...` to your directory where checkpoints of PCNO are stored.     
Change the data path in `PCNO/PCNO/2D_Flood/experiments_Flood_test.py` at Line 261: `path_test=...` to your directory containing the testing datasets.    
Run `PCNO/PCNO/2D_Flood/experiments_Flood_test.py` to predict spatiotemporal floods using PCNO.    
### DiffPCNO
Change the path in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` at Line 59: `results_path` to save the results.    
Change timesteps `T` and sample sizes (Line 66-88) in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` to the parameters in your downloaded checkpoints.   
Change the space sizes `Sy` and `Sx` (Line 151-165) in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` according to the flood scenario you want to predict.    
Change the path in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` at Line 221: `root=...` to your directory where checkpoints of DiffPCNO are stored.     
Change the path in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` at Line 250: `path_model_con=...` to your directory where checkpoints of PCNO are stored.     
Change the data path (Lines 257-260) in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` to your directory containing the testing flood datasets.        
Change testing timesteps in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` at Line 289: `T_test=...` to your testing scenario.  
Change the path in `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` at Line 676: `log_dir =...` to save the visualization results.    
Run `PCNO/DiffPCNO/2D_Flood/Flood_sampling.py` to obtain flood forecasting results `pred` and uncertainty results `pred_std` using DiffPCNO. 
## Atmospheric modeling
### PCNO
Change the path in `PCNO/PCNO/2D_Atmospheric/experiments_Atmospheric_test.py` at Line 57: `results_path` to save the results.      
Change the data path in `PCNO/PCNO/2D_Atmospheric/experiments_Atmospheric_test.py` at Line 59: `data_path` to your directory containing the atmospheric datasets.     
Change the path in `PCNO/PCNO/2D_Atmospheric/experiments_Atmospheric_test.py` at Line 187: `root=...` to your directory where checkpoints of PCNO are stored.         
Run `PCNO/PCNO/2D_Atmospheric/experiments_Atmospheric_test.py` to predict 2D gravity waves in atmospheric modeling using PCNO.      
### DiffPCNO
Change the path in `PCNO/DiffPCNO/2D_Atmospheric/Atmospheric_sampling.py` at Line 59: `results_path` to save the results.    
Change the data path in `PCNO/DiffPCNO/2D_Atmospheric/Atmospheric_sampling.py` at Line 62: `data_path` to your directory containing the atmospheric datasets.     
Change the path in `PCNO/DiffPCNO/2D_Atmospheric/Atmospheric_sampling.py` at Line 214: `root=...` to your directory where checkpoints of DiffPCNO are stored.     
Change the path in `PCNO/DiffPCNO/2D_Atmospheric/Atmospheric_sampling.py` at Line 241: `path_model_con=...` to your directory where checkpoints of PCNO are stored.           
Change the path in `PCNO/DiffPCNO/2D_Atmospheric/Atmospheric_sampling.py` at Line 800: `log_dir =...` to save the visualization and rollout MSE results.    
Run `PCNO/DiffPCNO/2D_Atmospheric/Atmospheric_sampling.py` to obtain atmospheric modeling results `pred` and uncertainty results `pred_std` using DiffPCNO. 

# 👻​ Training 
After configuring the file paths and parameters, run the following code for different spatiotemporal dynamics.  
## 1D Kuramoto–Sivashinsky dynamics
### PCNO
`python PCNO/PCNO/1D_KSE/experiments_fixed_viscosity_train_PCNO.py` for Kuramoto–Sivashinsky dynamics with fixed viscosity.    
`python PCNO/PCNO/1D_KSE/experiments_varying_viscosity_train_PCNO.py` for Kuramoto–Sivashinsky dynamics with varying viscosity.     
### DiffPCNO
`python PCNO/DiffPCNO/1D_KSE/experiments_fixed_viscosity_train_DiffPCNO.py` for Kuramoto–Sivashinsky dynamics with fixed viscosity.   
`python PCNO/DiffPCNO/1D_KSE/experiments_varying_viscosity_train_DiffPCNO.py` for Kuramoto–Sivashinsky dynamics with varying viscosity.   
### PCNO-Refiner
`python PCNO/PCNO-Refiner/experiments_1DKSE_fixed_viscosity_train_PCNO-Refiner.py` for Kuramoto–Sivashinsky dynamics with fixed viscosity.     
`python PCNO/PCNO-Refiner/experiments_1DKSE_varying_viscosity_train_PCNO-Refiner.py` for Kuramoto–Sivashinsky dynamics with varying viscosity.   
## 2D Kolmogorov turbulent flow
### PCNO
`python PCNO/PCNO/2D_Kolmogorov/experiments_Kolmogorov_PCNO.py` for Kolmogorov turbulent flow.      
### DiffPCNO
`python PCNO/DiffPCNO/2D_Kolmogorov/experiments_Kolmogorov_train_DiffPCNO.py` for Kolmogorov turbulent flow.    
## 2D Real-world flood forecasting
### PCNO
`python PCNO/PCNO/2D_Flood/experiments_Pakistan_train_PCNO.py` for low-fidelity Pakistan flood forecasting.   
`python PCNO/PCNO/2D_Flood/experiments_Australia_train_PCNO.py` for high-fidelity Australia flood forecasting.   
### DiffPCNO
`python PCNO/DiffPCNO/2D_Flood/experiments_Pakistan_train_DiffPCNO.py` for low-fidelity Pakistan flood forecasting.   
`python PCNO/DiffPCNO/2D_Flood/experiments_Australia_train_DiffPCNO.py` for high-fidelity Australia flood forecasting.   
## Atmospheric modeling
### PCNO
`python PCNO/PCNO/2D_Atmospheric/experiments_Atmospheric_train_PCNO.py` for 2D atmospheric modeling.     
`python PCNO/PCNO/3D_Atmospheric/experiments_Atmospheric_train_PCNO3D.py` for 3D atmospheric modeling.   
### DiffPCNO
`python PCNO/DiffPCNO/2D_Atmospheric/experiments_Atmospheric_train_DiffPCNO.py` for atmospheric modeling.   


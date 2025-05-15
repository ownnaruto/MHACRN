# MHACRN Traffic Forecasting

## Requirements

Install the following Python packages:

```bash
pip install numpy==1.22.4  
pip install pandas==1.3.5  
pip install eniops==0.4.0  
pip install torch==1.10.0  
pip install PyYAML==6.0 
```


## Datasets

Due to file size constraints, this repository includes only the METR‑LA dataset.
Other benchmark datasets (PEMS‑BAY, PEMS04, PEMS07, PEMS08) can be downloaded from their respective public sources.

## Reproduce Results 

To reproduce the METR‑LA experiments with the provided hyperparameters, run:

```bash
cd scripts 
python train.py -m mhacrn -g 0 
```

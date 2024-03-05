# PINN4NST - practice
Physics-informed neural networks parts for NST

### Conda Environment
```
conda create -n nst python=3.9

conda activate nst

pip install --upgrade pip

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

conda install -c conda-forge deepxde
```

### Naive PINN for Navier Stokes
- FeedForward Neural Networks with point-collocations
```
DDE_BACKEND=pytorch python pinns.py --config configs/baseline/Re500-pinns-05s.yaml
```

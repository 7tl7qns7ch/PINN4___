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
- FeedForward Neural Networks with pinn loss on pointwise collocations.
```
DDE_BACKEND=pytorch python pinns.py --config configs/baseline/Re500-pinns-05s.yaml
```

### Neural operator for Navier Stokes (Z. Li, et. al. 2022.)
- Fourier neural operator with pinn loss in spectral space.
```
python instance_opt.py --config configs/Re500-05s-test.yaml
```

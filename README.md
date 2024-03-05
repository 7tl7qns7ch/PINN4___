# PINN4NST - practice
Physics-informed neural network parts for NST. Codes are based on https://github.com/neuraloperator/physics_informed (Z. Li, et. al. 2022.) with small modifications for further research.

## Conda Environment
```
conda create -n nst python=3.9

conda activate nst

pip install --upgrade pip

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

conda install -c conda-forge deepxde
```

## Navier Stokes with Reynolds number 500
- spatial domain: $x\in (0, 2\pi)^2$
- temporal domain: $t \in \[0, 0.5\]$
- forcing: $-4\cos(4x_2)$
- Reynolds number: 500

Train set: data of shape (N, T, X, Y) where N is the number of instances, T is temporal resolution, X, Y are spatial resolutions. 
1. [NS_fft_Re500_T4000.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fft_Re500_T4000.npy) : 4000x64x64x65
2. [NS_fine_Re500_T128_part0.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fine_Re500_T128_part0.npy): 100x129x128x128
3. [NS_fine_Re500_T128_part1.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fine_Re500_T128_part1.npy): 100x129x128x128

Test set: data of shape (N, T, X, Y) where N is the number of instances, T is temporal resolution, X, Y are spatial resolutions. 
1. [NS_Re500_s256_T100_test.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_Re500_s256_T100_test.npy): 100x129x256x256    (**Download this 100 examples of NS**)
2. [NS_fine_Re500_T128_part2.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fine_Re500_T128_part2.npy): 100x129x128x128

Configuration file format: see `.yaml` files under folder `configs` for detail. 

## Naive PINN for Navier Stokes
- FeedForward Neural Networks (layers with [3, 100, 100, 100, 100, 3])with pinn loss on pointwise collocations.
```
DDE_BACKEND=pytorch python pinns.py --config configs/baseline/Re500-pinns-05s.yaml
```

## Neural operator for Navier Stokes
- Fourier neural operator with pinn loss in spectral space.
```
python instance_opt.py --config configs/Re500-05s-test.yaml
```

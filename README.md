# PINN4___
Physics-informed neural network parts for ___. Codes are based on https://github.com/neuraloperator/physics_informed (Z. Li, et. al. 2022.) and https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution (D. Shu, et. al. 2023) with some modifications for further research.

## Conda Environment
```
conda create -n pinn4 python=3.9

conda activate pinn4

pip install --upgrade pip

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

conda install -c conda-forge deepxde

pip install pyDOE
```
## Navier Stokes Data Generator
<p align="center">
<img src="https://github.com/7tl7qns7ch/PINN4___/assets/39257402/9aab4dcd-d051-49ba-ba6b-7c6e5489cfe9" width="250"> 
<img src="https://github.com/7tl7qns7ch/PINN4___/assets/39257402/8867da36-648d-4c95-a642-de7f156a066d" width="250"> 
<img src="https://github.com/7tl7qns7ch/PINN4___/assets/39257402/cd43b3d2-eb06-461b-a72d-229bbf86d67a" width="250">
</p>
```
NS_generator.ipynb (above figure can be obtained with changing args.Re=100, 1000, 10000 in the second code block)
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

**- Naive PINN for Navier Stokes**
- FeedForward Neural Networks (layers with [3, 50, 50, 50, 50, 50, 50, 3]) with pinn loss on pointwise collocations.
```
DDE_BACKEND=pytorch python pinns.py --config configs/baseline/Re500-pinns-05s.yaml
```

**- PINN + NO for Navier Stokes**
- Fourier neural operator with pinn loss in spectral space.
```
python instance_opt.py --config configs/Re500-05s-test.yaml --tqdm
```

## Navier Stokes with Reynolds number 1000
- spatial domain: $x\in (0, 2\pi)^2$
- temporal domain: $t \in \[0, 10\]$
- forcing: $-4\cos(4x_2) -0.1\omega(x, t)$
- Reynolds number: 1000

Test set: data of shape (N, T, X, Y) where N is the number of instances, T is temporal resolution, X, Y are spatial resolutions. 
1. (<a href="https://figshare.com/ndownloader/files/39181919">kf_2d_re1000_256_40seed</a>): 40x320x256x256    (**Download this 40 examples of NS**)

**- PINN + NO for Navier Stokes**
- Fourier neural operator with pinn loss in spectral space.
```
python train_pdeloss.py --tqdm
```

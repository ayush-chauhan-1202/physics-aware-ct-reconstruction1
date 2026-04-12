# 🩻 CT Reconstruction Toolkit

A comprehensive Python toolkit for Computed Tomography (CT) image reconstruction, implementing and comparing **Filtered Back Projection (FBP)**, **Iterative methods (SART/SIRT)**, and **Deep Learning (U-Net)** approaches.

---

## 📌 Project Overview

CT reconstruction transforms sinogram data (projections) back into cross-sectional images. This project benchmarks classical and modern methods side-by-side, providing modular, research-ready code with full evaluation metrics.

| Method | Speed | Noise Robustness | Accuracy |
|--------|-------|-----------------|----------|
| FBP    | ⚡ Fast | Low | Moderate |
| SART   | 🐢 Slow | High | High |
| SIRT   | 🐢 Slow | High | High |
| U-Net  | ⚡ Fast (inference) | Very High | Very High |

---

## 🗂️ Project Structure

```
ct_reconstruction/
├── data/
│   ├── raw/               # Raw sinogram / projection data
│   ├── processed/         # Preprocessed datasets
│   └── phantoms/          # Shepp-Logan and custom phantoms
├── src/
│   ├── algorithms/
│   │   ├── fbp.py         # Filtered Back Projection
│   │   ├── sart.py        # Simultaneous Algebraic Reconstruction
│   │   ├── sirt.py        # Simultaneous Iterative Reconstruction
│   │   └── unet.py        # U-Net deep learning model
│   ├── utils/
│   │   ├── data_loader.py # Data I/O utilities
│   │   ├── geometry.py    # CT geometry helpers
│   │   └── preprocessing.py
│   ├── visualization/
│   │   └── plotting.py    # Sinogram & image visualizations
│   └── evaluation/
│       └── metrics.py     # PSNR, SSIM, RMSE
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fbp_reconstruction.ipynb
│   ├── 03_iterative_methods.ipynb
│   └── 04_deep_learning.ipynb
├── results/
│   ├── figures/           # Output images & comparison plots
│   └── metrics/           # CSV metric reports
├── tests/
│   └── test_*.py
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/your-username/ct_reconstruction.git
cd ct_reconstruction

# Create environment
conda env create -f environment.yml
conda activate ct-recon

# Run FBP reconstruction on the Shepp-Logan phantom
python src/algorithms/fbp.py --phantom shepp_logan --angles 180

# Run iterative SART
python src/algorithms/sart.py --phantom shepp_logan --iterations 50

# Train / evaluate U-Net
python src/algorithms/unet.py --mode train --epochs 50
```

---

## 📊 Results

Evaluated on the Shepp-Logan phantom with 180 projection angles and Gaussian noise (σ=0.01):

| Method | PSNR (dB) | SSIM  | RMSE   |
|--------|-----------|-------|--------|
| FBP    | 28.4      | 0.821 | 0.038  |
| SART   | 33.7      | 0.912 | 0.021  |
| SIRT   | 32.9      | 0.905 | 0.023  |
| U-Net  | 37.2      | 0.954 | 0.014  |

> See `results/` and `notebooks/` for full comparison figures.

---

## 🔬 Methods

### Filtered Back Projection (FBP)
Classical analytic inversion using the Fourier Slice Theorem. Fast but sensitive to noise and sparse-view artefacts. Supports Ram-Lak, Shepp-Logan, and Hann filters.

### SART / SIRT
Algebraic iterative methods that model the forward projection as a linear system **Ax = b** and solve it iteratively. More robust to noise and limited-angle scenarios.

### U-Net
Encoder–decoder convolutional network trained end-to-end to map noisy/sparse FBP reconstructions to clean ground-truth images. Achieves state-of-the-art quality at inference speed.

---

## 📦 Dependencies

- Python ≥ 3.9
- NumPy, SciPy, scikit-image
- PyTorch ≥ 2.0
- ASTRA Toolbox (optional GPU projector)
- matplotlib, seaborn

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome!

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

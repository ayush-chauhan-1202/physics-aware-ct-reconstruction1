# 🩻 CT Reconstruction Toolkit — Industrial Edition

A research-ready Python toolkit for industrial CT image reconstruction. Simulates a realistic multi-material cylindrical phantom with six defect classes, applies a physically accurate noise pipeline, and benchmarks FBP, SART, and SIRT side-by-side with a single command.

---

## 🚀 Quickstart (local)

```bash
# 1. Clone
git clone https://github.com/your-username/ct_reconstruction.git
cd ct_reconstruction

# 2. Create environment (conda recommended)
conda env create -f environment.yml
conda activate ct-recon

# OR with pip
pip install -r requirements.txt

# 3. Run the full comparison pipeline (default: 384px, 180 angles, medium noise)
python run_comparison.py

# 4. View outputs
ls results/          # phantom.npy, fbp.npy, sart.npy, sirt.npy, metrics.csv
ls results/figures/  # comparison.png
```

---

## ⚙️  run_comparison.py — All Options

```
python run_comparison.py [OPTIONS]

  --size        INT     Phantom side length in pixels         [384]
  --angles      INT     Number of projection angles           [180]
  --noise       STR     Noise preset: none/low/medium/high    [medium]
  --fbp-filter  STR     FBP ramp filter                       [shepp-logan]
  --sart-iter   INT     SART iterations                       [50]
  --sirt-iter   INT     SIRT iterations                       [80]
  --sart-relax  FLOAT   SART relaxation lambda                [0.9]
  --sirt-relax  FLOAT   SIRT relaxation lambda                [1.0]
  --seed        INT     Global RNG seed                       [42]
  --out-dir     PATH    Output directory                       [results]
  --fbp-only            Run FBP only (fast smoke test)
  --no-figure           Skip figure generation
```

### Example runs

```bash
# High noise, more iterations, bigger phantom
python run_comparison.py --size 512 --angles 360 --noise high \
    --sart-iter 80 --sirt-iter 120 --out-dir results/high_noise

# Fast FBP-only check
python run_comparison.py --fbp-only --no-figure

# Reproducible custom seed
python run_comparison.py --seed 99 --out-dir results/seed99
```

---

## 🏗️  Phantom Description

```
Material layers (μ, normalised):
  1.00  Tungsten-carbide core pin
  0.85  Stainless-steel inner sleeve
  0.55  Aluminium-7075 alloy body (with radial corrosion gradient)
  0.18  Polymer / CFRP composite shell
  0.00  Air background

Defects:
  • 3 spherical voids        (large 12px / medium 8px / small 5px)
  • 55 micro-pores           at steel/Al interface  (1–3 px each)
  • Hairline crack           2-px diagonal in Al zone
  • Dense inclusion          high-μ ellipse (Pb fragment, μ=0.95)
  • Delamination arc         thin air gap at polymer/Al boundary
  • Density-gradient zone    radial corrosion in Al body
```

---

## 🔊  Noise Pipeline

Four presets (configured in `run_comparison.py`):

| Preset | I₀ (photons) | σ_e | Ring strength | Beam hardening |
|--------|-------------|-----|---------------|----------------|
| none   | 10⁹         | 0   | 0             | 0              |
| low    | 50 000      | 0.002 | 0.5 %       | 0 %            |
| medium | 10 000      | 0.005 | 1.0 %       | 3 %            |
| high   | 2 000       | 0.015 | 2.0 %       | 6 %            |

Applied in order: beam hardening → Poisson → Gaussian → rings.

---

## 🗂️ Project Structure

```
ct_reconstruction/
├── run_comparison.py             ← ONE-COMMAND PIPELINE ENTRY POINT
├── src/
│   ├── algorithms/
│   │   ├── fbp.py               FBP (ramp/shepp-logan/hann/cosine filters)
│   │   ├── sart.py              SART with tqdm progress
│   │   ├── sirt.py              SIRT with tqdm progress
│   │   └── unet.py              U-Net (training + inference)
│   ├── utils/
│   │   ├── industrial_phantom.py  ← PhantomConfig + build_industrial_phantom()
│   │   ├── preprocessing.py       ← NoiseConfig + apply_realistic_noise()
│   │   ├── data_loader.py         load/save sinograms, results, DICOM
│   │   └── geometry.py            parallel/fan-beam geometry helpers
│   ├── visualization/
│   │   ├── comparison_figure.py  ← 5-row publication figure
│   │   └── plotting.py           sinogram, error map, metric bar helpers
│   └── evaluation/
│       └── metrics.py            PSNR, SSIM, RMSE, MAE, compare_methods()
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fbp_reconstruction.ipynb
│   ├── 03_iterative_methods.ipynb
│   └── 04_deep_learning.ipynb
├── results/                      (created on first run)
│   ├── phantom.npy
│   ├── fbp.npy / fbp.json
│   ├── sart.npy / sart.json
│   ├── sirt.npy / sirt.json
│   ├── metrics.csv
│   └── figures/comparison.png
├── tests/test_all.py
├── requirements.txt
├── environment.yml
└── setup.py
```

---

## 📦 Dependencies

```
numpy >= 1.24
scipy >= 1.10
scikit-image >= 0.21
matplotlib >= 3.7
tqdm >= 4.65
torch >= 2.0        (U-Net only)
pandas >= 2.0
h5py >= 3.9
pydicom >= 2.4      (DICOM loading only)
```

---

## 🔬 Running Individual Algorithms

```bash
# FBP only (also generates sinogram from scratch)
python -m src.algorithms.fbp --phantom industrial --noise medium --filter shepp-logan

# SART
python -m src.algorithms.sart --iterations 60 --noise high

# SIRT
python -m src.algorithms.sirt --iterations 100 --relaxation 0.8

# Run tests
pytest tests/ -v
```

---

## 📄 License

MIT — see LICENSE.

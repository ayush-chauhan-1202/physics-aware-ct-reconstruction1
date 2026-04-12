# 🧠 Physics-Aware CT Simulation & Industrial Defect Modeling

> A production-grade, physics-informed pipeline for generating realistic cone-beam CT datasets with industrial defects and scanner artefacts.

---

## 🚀 Why This Project Matters

Modern CT-based inspection systems rely heavily on **data-driven models**, yet real labeled data is scarce, expensive, and often proprietary.

This project bridges that gap by:
- Simulating **high-fidelity CT physics**
- Generating **label-rich industrial defect datasets**
- Enabling **ML + reconstruction research in controlled environments**

---

## 🏗️ End-to-End Pipeline

```
3D Phantom (μ)
   ↓
Cone-beam Projection
   ↓
Beer–Lambert Physics
   ↓
Noise + Artefacts
   ↓
Detector Measurements
```

---

## 📸 Sample Outputs (Add Images Here)
<img width="128" height="128" alt="image" src="https://github.com/user-attachments/assets/5414b1ab-f406-46ed-a6f1-84e1b2beec23" /> Phantom

 ![clean_projections_180x90x180_float32](https://github.com/user-attachments/assets/34664578-3742-46ef-aa6a-d7e9fde52572) Clean Projections

![noisy_projections_180x90x180_float32](https://github.com/user-attachments/assets/28bea6a4-1fe0-4657-b1a6-e80a8844e5c1)
Noisy Projections


> Add images like:
- Phantom slices
- Clean vs noisy projections
- Sinograms

```
/assets/phantom.png
/assets/projection.png
/assets/noise_comparison.png
```

---

## ⚙️ CT Geometry

| Parameter | Description |
|----------|------------|
| Nx, Ny, Nz | Volume resolution |
| SID | Source → Isocenter |
| SDD | Source → Detector |
| n_angles | Number of projections |
| det_rows, det_cols | Detector resolution |

---

## 🧱 Industrial Phantom

Multi-material simulation:
- CFRP (base)
- Aluminium / Steel layers
- Tungsten / Lead inclusions

---

## 🧨 Defect Library

| Defect | Description |
|------|------------|
| Pores | Random spherical voids |
| Micro-porosity | Clustered defects |
| Cracks | Tilted ellipsoids |
| Inclusion | Dense foreign material |
| Delamination | Thin planar separation |
| Corrosion | Smooth degradation |

---

## 🔊 Noise & Artefacts (Physics-Aware)

| Effect | Model |
|------|------|
| Beam Hardening | p' = p + αp² |
| Poisson | Photon statistics |
| Gaussian | Electronics noise |
| Rings | Detector bias |
| Scatter | Low-frequency blur |

---

## 📦 Outputs

| File | Description |
|-----|------------|
| phantom.npy | Ground truth |
| projections_clean.npy | Ideal |
| projections_noisy.npy | Realistic |
| geometry.json | Scanner config |

---

## 🧪 Experiment Tracking

### Variant: ___________________

| Parameter | Value |
|----------|------|
| Noise |  |
| α |  |
| σ |  |
| Scatter |  |

Observations:
- 
- 

---

## 📊 Applications

- CT denoising (DL models)
- Defect detection (classification/segmentation)
- Sparse-view reconstruction
- Physics-aware generative models
- Sim-to-real domain adaptation

---

## 🧠 Key Technical Highlights

- Physics-consistent forward model
- Signal-dependent noise (Poisson)
- Industrial defect realism
- Modular pipeline (extensible)

---

## 🔮 Roadmap

- Polychromatic modeling μ(E)
- Detector PSF
- Focal spot blur
- Learned noise models
- Real-data calibration

---

## 👤 Author

**Ayush Chauhan**  
Applied ML | CT Reconstruction | Industrial AI  

---

## ⭐ If you find this useful

Star ⭐ the repo and feel free to connect!

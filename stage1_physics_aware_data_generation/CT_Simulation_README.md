# 🧠 Physics-Aware CT Data Simulation Pipeline

## 🚀 Overview
A modular, physics-informed pipeline for generating synthetic cone-beam CT datasets with realistic industrial geometries, defects, and noise models.

Designed for:
- Machine Learning (denoising, defect detection, self-supervised learning)
- CT Reconstruction (FDK, MBIR)
- Industrial Inspection (NDE workflows)

---

## 🏗️ System Pipeline

3D Phantom (μ volume)  
→ Cone-beam projection  
→ Beer–Lambert law  
→ Noise & artefacts  
→ Final detector measurements  

---

## ⚙️ CT Geometry

| Parameter | Description |
|----------|------------|
| Nx, Ny, Nz | Volume resolution |
| SID | Source-to-isocenter distance |
| SDD | Source-to-detector distance |
| n_angles | Number of projections |
| det_rows, det_cols | Detector resolution |

Forward model:

I = I0 * exp(-∫ μ dl)

---

## 🧱 Phantom & Materials

Multi-material industrial phantom with:
- CFRP base structure
- Aluminium / Steel regions
- High-density inclusions (Tungsten/Lead)

---

## 🧨 Defect Modeling

- Spherical pores (voids)
- Micro-porosity clusters
- Tilted cracks (ellipsoidal)
- Dense inclusions
- Delamination layers
- Corrosion gradients

---

## 🔊 Noise & Artefacts

### Beam Hardening
p' = p + α p²

### Poisson Noise
I ~ Poisson(I)

### Gaussian Noise
I += N(0, σ²)

### Ring Artefacts
Detector column bias

### Scatter
I += blur(I) * β

---

## 📦 Outputs

| File | Description |
|-----|------------|
| phantom.npy | Ground truth volume |
| projections_clean.npy | Ideal projections |
| projections_noisy.npy | Noisy projections |
| geometry.json | Scanner config |

---

## 🧪 Experimental Variants

### Variant 1: ___________________

Parameters:
- Noise Level:
- Beam Hardening α:
- Photon Count:
- Gaussian σ:
- Scatter β:
- Smoothing:

Observations:
- 
- 
- 

---

### Variant 2: ___________________

Parameters:
- 

Observations:
- 

---

### Variant 3: ___________________

Parameters:
- 

Observations:
- 

---

## 📊 Applications

- ML-based denoising & defect detection
- Sparse-view reconstruction
- Physics-aware AI models
- Simulation-to-real domain adaptation

---

## 🔮 Future Work

- Polychromatic modeling μ(E)
- Detector PSF / blur
- Focal spot modeling
- Advanced scatter simulation

---

## 👤 Author

Ayush Chauhan  
Applied ML | CT Reconstruction | Industrial AI

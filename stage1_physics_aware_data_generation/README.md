# Stage 1 — 3D Industrial Phantom & Cone-Beam Projections

## Quickstart

```bash
pip install -r requirements.txt

# Default run  (128×128×64 volume, 180 angles, medium noise)
python run_stage1.py

# Custom
python run_stage1.py --Nx 128 --Ny 128 --Nz 64 \
    --n-angles 180 --noise medium \
    --SID 500 --SDD 900 --seed 42

# Fast test (reduced ray samples, no figures)
python run_stage1.py --fast --no-figure
```

## Outputs

```
results/
├── volume/
│   ├── phantom.npy          3D attenuation volume (Nz×Ny×Nx, float32)
│   └── phantom_meta.json    physical μ values, voxel size
├── projections/
│   ├── clean.npy            (n_angles × det_rows × det_cols)
│   ├── noisy.npy            with Poisson + Gaussian + rings + BH
│   └── geometry.json        scanner parameters
└── figures/
    ├── stage1_summary.png   ← main figure, start here
    ├── orthogonal.png
    ├── defect_panel.png
    ├── projection.png
    ├── sinogram_stack.png
    └── geometry.png
```

## Phantom Materials (at 100 keV)

| Material      | μ (cm⁻¹) | μ (normalised) |
|---------------|----------|----------------|
| Air / defects | 0.000    | 0.000          |
| CFRP polymer  | 0.180    | 0.142          |
| Aluminium     | 0.461    | 0.363          |
| Stainless steel | 0.797  | 0.627          |
| Tungsten carbide | 1.270 | 1.000          |
| Lead inclusion | 1.100   | 0.866          |

## Defects Modelled

- **Spherical pores** (4 sizes) — gas-filled voids at varying depths
- **Micro-porosity cluster** — 80 tiny pores at the steel/Al interface
- **Tilted crack** — ellipsoidal planar void, tapers axially
- **Dense inclusion** — ellipsoidal lead fragment
- **Delamination arc** — thin air shell at polymer/Al boundary
- **Corrosion gradient** — radial μ drop in Al, stronger at one end

## Noise Pipeline

```
beam hardening → Poisson → Gaussian → ring artefacts → scatter
```

| Preset    | I₀ photons | σ_e   | Rings | BH coeff |
|-----------|-----------|-------|-------|----------|
| noiseless | 10⁹       | 0     | 0     | 0        |
| low       | 50 000    | 0.002 | 0.5%  | 1%       |
| medium    | 10 000    | 0.005 | 1.0%  | 3%       |
| high      | 2 000     | 0.015 | 2.0%  | 6%       |

## Next Stage

Stage 2 will load `projections/noisy.npy` + `geometry.json` and implement
FDK reconstruction with ramp filtering, back-projection across cone angles,
slice-wise metrics, and comparison figures.

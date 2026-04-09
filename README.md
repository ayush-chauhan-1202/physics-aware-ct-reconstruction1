# physics-aware-ct-reconstruction1

Industrial X-ray CT is a critical tool for non-destructive evaluation (NDE) in aerospace and manufacturing, enabling inspection of internal defects such as porosity, cracks, and coating inconsistencies. In real-world deployments, acquisition is often constrained by limited projection views, low-dose requirements, and system noise, leading to reconstruction artifacts that degrade inspection reliability.

This project develops a physics-informed reconstruction pipeline that integrates analytical methods (e.g., FDK/SIRT) with deep learning-based refinement to improve reconstruction quality under sparse and noisy acquisition conditions. The approach enforces consistency with the underlying imaging physics while leveraging learned priors to enhance structural fidelity and artifact suppression.


## Evaluation Metrics

We evaluate reconstruction quality using a combination of pixel-level, structural, and task-based metrics:

- PSNR and SSIM to measure reconstruction fidelity and structural similarity
- MAE to quantify voxel-wise reconstruction error
- Edge sharpness (gradient-based) to assess preservation of fine structures
- Task-based proxy metric to evaluate defect visibility under different reconstruction methods

This multi-level evaluation ensures that improvements are not only numerical but also meaningful for downstream inspection tasks.

# R_VPL-SMA connectivity analysis code

Version **1.0.0** · released 2026-10-26

This repository contains the study-specific analysis and verification code supporting:

> **Pretreatment thalamic-premotor connectivity and recovery of consciousness in patients receiving short-term spinal cord stimulation**

The repository covers the predefined ROI/connectivity analyses, focused R_VPL-SMA models,
resampling and statistical validation, preprocessing-provenance sensitivity analyses, and
publication-figure generation.

## Scope

This is a **research reproducibility repository**, not a clinical prediction tool.

The study is a retrospective single-centre cohort of patients with prolonged disorders of
consciousness who received short-term spinal cord stimulation (stSCS). The code evaluates
associations between pretreatment functional connectivity and six-month recovery outcomes.
It does not estimate a causal treatment effect or a stimulation-specific treatment-response
effect.

## What is included

- `src/primary/` — archived core analysis algorithms used for connectivity and focused-edge analyses.
- `src/statistics/` — Firth-logistic, residual-permutation, calibration, max-statistic and broader-connectivity validation code.
- `src/sensitivity/` — preprocessing-provenance sensitivity analysis.
- `src/verification/` — numerical and reconstruction checks.
- `src/figures/` — scripts used to construct or verify publication figures.
- `metadata/` — software environment and source-integrity manifest.
- `docs/` — data requirements, reproducibility notes and release instructions.

## What is **not** included

This repository intentionally contains **no patient MRI, participant-level clinical data,
patient identifiers, or patient-level linkage tables**. Those data are controlled by
institutional ethics and privacy requirements.

The public code package also does not redistribute external software or atlases, including
FSL, fMRIPrep, DPABI, BrainNet Viewer, or the Julich-Brain atlas.

The upstream clinical MRI preprocessing itself is not reproduced here. In the retrospective
archive, fMRIPrep was used when a suitable high-resolution T1-weighted structural image was
available and DPABI was used otherwise; all patient datasets subsequently entered a common
lesion-aware downstream workflow. The public repository contains the study-specific
downstream analysis, verification and sensitivity code.

## Data sources

The healthy-reference dataset used only for feature scaling is publicly available from
OpenNeuro as **ds006623**.

Patient-level inputs required to rerun the complete analysis are not public. Researchers with
approved access should follow `docs/DATA_REQUIREMENTS.md` and place controlled inputs outside
version control.

## Software environment

The validated analysis environment used Python 3.13.13 with the principal package versions
recorded in `metadata/environment.json`. Install the Python dependencies with:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

FSL is required for the lesion-aware ICA workflow. The optional compiled Firth accelerator is
built from `src/statistics/firth_accelerator.cpp`. Example commands:

```bash
# Linux
g++ -O3 -fopenmp -shared -fPIC src/statistics/firth_accelerator.cpp -o bin/libfirth_accelerator.so

# Windows (Rtools/MinGW)
g++ -O3 -fopenmp -shared src/statistics/firth_accelerator.cpp -o bin/firth_accelerator.dll
```

Before long resampling runs, verify numerical agreement between the compiled accelerator and
the Python Firth implementation.

## Configuration

Scripts use relative paths and/or environment variables rather than workstation-specific
absolute paths. See `.env.example` and `docs/DATA_REQUIREMENTS.md`.

Common variables include:

- `RVPL_SMA_PROJECT_ROOT`
- `RVPL_SMA_DISCOVERY_DIR`
- `RVPL_SMA_SUPPLEMENT_INPUTS_DIR`
- `RVPL_SMA_PRIOR_SUBMISSION_DIR`
- `RVPL_SMA_VALIDATION_OUTPUT`
- `RVPL_SMA_VERIFICATION_OUTPUT`
- `RVPL_SMA_SENSITIVITY_OUTPUT`
- `RVPL_SMA_RESTRICTED_INPUTS`

## Reproducibility

Because participant-level clinical and MRI data cannot be publicly redistributed, the complete
patient-level workflow cannot be executed from this repository alone. The repository is
intended to make the study-specific algorithms, statistical procedures, parameter choices,
figure-generation code and numerical-verification logic inspectable and reusable.

See:

- `docs/REPRODUCIBILITY.md`
- `docs/DATA_REQUIREMENTS.md`
- `metadata/source_integrity_manifest.tsv`
- `metadata/PUBLICATION_QC_REPORT.md`

## Release integrity

`SHA256SUMS.txt` contains SHA256 digests for the files in this release. The source-integrity
manifest links the cleaned public scripts to the staged publication-candidate sources.

The public packaging step changed repository organization, configuration and privacy-sensitive
I/O behavior where required for safe release. It did not intentionally change the underlying
statistical methods.

## Citation

If you use this code, please cite the associated article and the archived software release.
Machine-readable citation metadata are supplied in `CITATION.cff`.

A Zenodo DOI will be added after the GitHub release is archived.

## License

This code is released under the MIT License. See `LICENSE`.

## Contact

For questions about the study, data access or code, please contact the corresponding authors
listed in the associated article.

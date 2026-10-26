# Reproducibility map

This release separates scientific code from restricted participant-level inputs.

## Core analysis

`src/primary/clinical_models.py`
implements the Firth penalized-logistic routines and associated clinical-model utilities used
by the focused connectivity analyses.

`src/primary/connectivity_matrix_pipeline.py` and
`src/primary/seed_definition_pipeline.py` preserve the core connectivity / seed-processing
algorithms used in the archived workflow.

`src/primary/focused_edge_validation.py` preserves the focused R_VPL-SMA validation logic.

## Statistical validation

`src/statistics/statistical_validation.py` implements the validated fixed-edge statistical
workflow, including:

- uniform 5,000-iteration residual-permutation analyses;
- fixed-design null calibration;
- candidate-conditioned anatomical comparison sets;
- complete-case sensitivity;
- synchronized max-|Welch t| whole-connectome sensitivity;
- broader-connectivity sensitivity;
- spatial-cohort auditing.

`src/statistics/firth_accelerator.cpp` is an optional numerical accelerator. The accelerator
must be compiled for the local platform and numerically checked against the Python Firth
implementation before long resampling runs.

The analysis specification is in `src/statistics/ANALYSIS_SPECIFICATION.md`.

## Preprocessing-provenance sensitivity

`src/sensitivity/preprocessing_provenance_sensitivity.py` repeats the primary model after
adding the recorded upstream pipeline label (DPABI versus fMRIPrep) while preserving the
original primary cohort, outcome, feature scale and clinical covariates.

The public version deliberately does not write a patient-level linkage CSV.

## Verification

`src/verification/` contains reconstruction and numerical checks used to verify the analysis
outputs. These scripts require approved local inputs and are not independent patient
replications.

## Figures

- Figure 1: `src/figures/figure1_outcome_guided.py`
- Robustness/anatomical comparison figure: `src/figures/figure3_robustness_anatomical.py`
- SMA/6mp spatial localisation: `src/figures/figure4_sma6mp_spatial.py`
- Supporting/publication adapters: other scripts in `src/figures/`

## Inferential boundary

The R_VPL-SMA connection was selected in an outcome-guided 435-edge screen in the same
treated cohort. Subsequent fixed-edge analyses characterize robustness and anatomical/spatial
context; they are not independent patient validation and do not establish a causal or
stimulation-specific treatment effect.

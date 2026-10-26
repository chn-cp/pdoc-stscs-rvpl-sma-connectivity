# Data requirements and privacy

## Publicly available data

The healthy-reference dataset used for numerical feature scaling is available from OpenNeuro
as **ds006623**.

## Restricted patient-level inputs

The complete rerun of the study requires institutionally controlled patient-level inputs,
including the following classes of files:

1. Pretreatment functional-connectivity matrices and node metadata.
2. Deidentified clinical covariates and six-month CRS-R/diagnostic outcomes.
3. Imaging-QC records and preprocessing-provenance labels.
4. Lesion masks / lesion-support summaries used to determine ROI estimability.
5. Seed-to-voxel statistical inputs for the spatial analyses.

These data are **not** part of the public repository because MRI and individual-level clinical
information may be identifiable and are governed by institutional ethics and privacy
requirements.

## Expected local layout

Scripts accept environment-variable overrides (see `.env.example`). A typical controlled
layout is:

```text
restricted_inputs/
├── discovery_analysis/
│   ├── 00_inputs/
│   ├── 02_spatial_convergence/
│   └── 05_master_results/
├── supplement_inputs/
├── statistical_validation/
├── spatial_convergence/
└── prior_submission/
```

The exact institutional archive does not need to be copied into the repository. Point the
environment variables to approved local locations instead.

## Important

- Never commit patient-level linkage tables or clinical identifiers.
- Never upload raw/processed patient MRI to this GitHub/Zenodo release unless separately
  approved for public sharing.
- Do not infer missing demographic variables from filenames or study location.
- Aggregate Source Data and Supplementary Data distributed with the article are separate
  from the restricted participant-level inputs.

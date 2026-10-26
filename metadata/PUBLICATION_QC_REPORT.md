# Publication QC report

Release: v1.0.0
Date: 2026-10-26

## Automated checks

- Python source files compiled successfully: **14 / 14**
- C++ accelerator syntax check passed with `g++ -std=c++17 -fsyntax-only -fopenmp`
- Internal manuscript-version labels: **0**
- Internal workflow-development labels: **0**
- Workstation/server absolute paths: **0**
- Credential-like strings (API keys, tokens, passwords): **0**
- Direct email addresses: **0**
- Included patient MRI / clinical data files: **0**
- Included patient-level linkage tables: **0**

The only tabular file in `metadata/` is a source-integrity manifest containing file names and
SHA256 hashes, not participant data.

## Deliberate publication-safety changes

- Public-facing file and directory names use scientific function rather than internal version labels.
- Absolute workstation paths were replaced by relative/configurable paths or environment variables.
- The preprocessing-provenance sensitivity does not write its former patient-level linkage CSV.
- Participant-level common-sample membership and spatial-audit rows are not written by the public statistical-validation script.
- Manuscript-build/author-review helper code was excluded because it is not part of the scientific analysis.

## Known reproducibility constraints

- Participant-level patient inputs are restricted and are not included.
- FSL release numbers used in the archived preprocessing workflow were not recovered.
- Exact historical versions of scikit-learn and PyMuPDF were not retained in the analysis metadata; they are therefore unpinned in `requirements.txt`.
- The optional C++ Firth accelerator requires local compilation and numerical cross-checking against the Python implementation.

## Human checks before DOI minting

1. Confirm the creator list in `CITATION.cff` and `.zenodo.json`.
2. Confirm MIT is the intended licence under institutional policy.
3. Confirm the GitHub repository contains only the contents of this release.
4. Create GitHub release `v1.0.0`, archive it in Zenodo, and use the assigned DOI in the manuscript and Reporting Summary.

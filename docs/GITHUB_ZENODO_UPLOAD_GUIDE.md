# GitHub + Zenodo release guide

This package is prepared for release **v1.0.0**.

## 1. Create the GitHub repository

Recommended repository name:

`pdoc-stscs-rvpl-sma-connectivity`

Create a new empty repository. Public visibility can be enabled immediately or after your
final visual review.

## 2. Upload this package

Upload the **contents of this folder**, not the outer ZIP itself as the only repository file.

The GitHub repository root should directly contain:

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `.zenodo.json`
- `requirements.txt`
- `environment.yml`
- `src/`
- `docs/`
- `metadata/`

Do **not** upload restricted patient data.

## 3. Connect GitHub to Zenodo

Sign in to Zenodo, connect the GitHub integration, and enable this repository for archiving.

## 4. Create the release

On GitHub create a release/tag:

`v1.0.0`

Suggested release title:

`R_VPL-SMA connectivity analysis code v1.0.0`

Zenodo will archive the GitHub release and assign a DOI.

## 5. Final manuscript updates

After Zenodo assigns the DOI:

- replace the provisional Code Availability statement with the final Zenodo DOI;
- enter the same DOI in the Nature Portfolio Reporting Summary;
- cite the software DOI where appropriate.

Suggested final Code Availability wording:

> Study-specific Python and C++ code used for connectivity analyses, statistical modelling,
> resampling procedures and figure generation is publicly available at Zenodo
> (DOI: **[insert Zenodo DOI]**) under the MIT License.

## 6. Final safety check

Before making the repository public, confirm that it contains no:

- patient MRI;
- participant-level clinical data;
- linkage tables;
- names, medical-record numbers or other identifiers;
- API keys, passwords or tokens;
- workstation-specific absolute paths.

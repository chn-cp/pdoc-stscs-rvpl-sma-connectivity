# Release notes — v1.0.0

Initial public code release supporting the manuscript:

**Pretreatment thalamic-premotor connectivity and recovery of consciousness in patients receiving short-term spinal cord stimulation**

## Included

- core connectivity and focused-edge analysis algorithms;
- Firth-logistic and resampling code;
- fixed-design calibration and multiplicity sensitivity code;
- preprocessing-provenance sensitivity;
- numerical/reconstruction verification scripts;
- publication-figure generation code;
- software-environment and source-integrity metadata.

## Publication-safety changes relative to the internal staging package

These changes do **not** alter the statistical algorithms:

- public-facing files and directories were renamed by scientific function;
- workstation-specific absolute paths were replaced by configurable relative paths or
  environment variables;
- internal manuscript-version terminology was removed from public-facing code and metadata;
- the preprocessing-provenance sensitivity no longer writes a patient-level linkage CSV;
- author-review and manuscript-build helper code that is not part of the scientific analysis
  was excluded;
- documentation, licence, citation metadata and privacy guidance were added.

Restricted patient inputs remain excluded.

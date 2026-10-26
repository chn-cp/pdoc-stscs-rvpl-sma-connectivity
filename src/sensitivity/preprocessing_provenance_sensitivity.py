"""Preprocessing-provenance sensitivity for the primary R_VPL-SMA association.

This script preserves the primary cohort, outcome, feature scaling, clinical covariates,
Firth estimator and residual-permutation algorithm. Patient-level inputs are restricted
and are not distributed in this repository.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "10")
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import ctypes, hashlib, json, platform
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("RVPL_SMA_PROJECT_ROOT", Path.cwd()))
OUT = Path(os.environ.get("RVPL_SMA_SENSITIVITY_OUTPUT", ROOT / "outputs" / "preprocessing_provenance"))
MASTER = Path(os.environ.get("RVPL_SMA_MASTER_COHORT", ROOT / "restricted_inputs" / "discovery_analysis" / "00_inputs" / "master_imaging_cohort_n30.csv"))
LEDGER = Path(os.environ.get("RVPL_SMA_QC_LEDGER", ROOT / "restricted_inputs" / "supplement_inputs" / "qc33_deidentified.csv"))
BASELINE_RESULTS = Path(os.environ.get("RVPL_SMA_BASELINE_RESULTS", ROOT / "restricted_inputs" / "statistical_validation" / "robustness_uniform5000.csv"))
ACCELERATOR = Path(os.environ.get("FIRTH_ACCELERATOR_LIB", ROOT / "bin" / ("firth_accelerator.dll" if os.name == "nt" else "libfirth_accelerator.so")))

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT / "primary"))
import clinical_models as core

SEED = 2026102301
B = 5000
FEATURE = "R_VPL_SMA_z_local26_z"
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2,
                   default=lambda v: v.item() if hasattr(v, "item") else str(v)),
        encoding="utf-8",
    )

dump("ANALYSIS_PLAN.json", dict(
    question="Does the focused association remain after adjustment for recorded upstream preprocessing provenance?",
    analysis_status="Additional sensitivity analysis; not part of the original outcome-guided screen",
    cohort="Unchanged primary focused-edge n=26; no imputation or feature reselection",
    outcome="Six-month MCS+/EMCS attainment",
    predictor=FEATURE,
    feature_unit="One unchanged healthy-reference SD",
    base_covariates=["baseline CRS-R", "log1p injury-to-stimulation days"],
    additional_covariate="fMRIPrep=1; DPABI=0, from archived QC ledger",
    estimator="Firth penalized logistic regression",
    inference="Wald 95% CI and two-sided Wald P; 5000 predictor-residual permutations with provenance in the nuisance design, absolute edge coefficient, plus-one rule",
    permutations=B, seed=SEED,
    scope="No interaction testing, subgroup significance testing, feature reselection, preprocessing changes or family-correction recalculation",
))

data = pd.read_csv(MASTER)
ledger = pd.read_csv(LEDGER)
assert data.Subject.is_unique and ledger.Subject.is_unique
data = data.merge(ledger[["Subject", "pipeline"]], on="Subject", how="left", validate="one_to_one")
data = data[np.isfinite(data[FEATURE])].copy()
assert len(data) == 26 and data.favourable6.sum() == 12 and data.primary_available.all()
assert data.pipeline.notna().all() and set(data.pipeline) == {"DPABI", "fMRIPrep"}
assert data[["CRS_pre", "days_to_surgery", "log1p_days", "favourable6", FEATURE]].notna().all().all()
assert np.allclose(np.log1p(data.days_to_surgery), data.log1p_days, rtol=1e-12)
data["fMRIPrep_indicator"] = (data.pipeline == "fMRIPrep").astype(int)

z = lambda x: np.asarray((x - x.mean()) / x.std(ddof=1), float)
y = np.ascontiguousarray(data.favourable6.to_numpy(float))
x = np.ascontiguousarray(data[FEATURE].to_numpy(float))
Z0 = np.column_stack([np.ones(len(data)), z(data.CRS_pre), z(data.log1p_days)])
Z = np.ascontiguousarray(np.column_stack([Z0, data.fMRIPrep_indicator]))
X = np.ascontiguousarray(np.column_stack([Z, x]))

base = core.firth_fit(np.column_stack([Z0, x]), y)
fit = core.firth_fit(X, y)
assert base["converged"] and fit["converged"]
assert np.linalg.matrix_rank(X) == X.shape[1]

def summary(fit, index=-1):
    b = float(fit["beta"][index]); s = float(fit["se"][index])
    return dict(
        n=len(data), events=int(y.sum()), beta=b, SE=s, OR=float(np.exp(b)),
        CI95_low=float(np.exp(b - 1.96*s)), CI95_high=float(np.exp(b + 1.96*s)),
        Wald_P=float(2*stats.norm.sf(abs(b/s))),
        converged=bool(fit["converged"]), iterations=int(fit["iterations"]),
    )

r0 = summary(base)
r = summary(fit)
old = pd.read_csv(BASELINE_RESULTS).iloc[0]
assert np.isclose(r0["OR"], old.OR, rtol=1e-8)
r0["residual_permutation_P"] = float(old.permutation_p)
r0["permutations"] = int(old.permutations)
r0["permutation_source"] = "Validated primary-model result, reused without rerunning the random stream"

if os.name == "nt":
    rtools = os.environ.get("RTOOLS_BIN")
    dll_ctx = os.add_dll_directory(rtools) if rtools and hasattr(os, "add_dll_directory") else None
lib = ctypes.CDLL(str(ACCELERATOR))
D = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
I = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
lib.fit_one.argtypes = [D,D,ctypes.c_int,ctypes.c_int,D,D,I]
lib.permute.argtypes = [D,D,D,D,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,D,I]

cb = np.zeros(X.shape[1]); cs = cb.copy(); co = np.zeros(1, np.int32)
lib.fit_one(X, y, len(y), X.shape[1], cb, cs, co)
assert co[0] == 1 and np.max(abs(cb-fit["beta"])) < 1e-6 and np.max(abs(cs-fit["se"])) < 1e-6

fitted = np.ascontiguousarray(Z @ np.linalg.lstsq(Z, x, rcond=None)[0])
resid = np.ascontiguousarray(x - fitted)
null = np.zeros(B); status = np.zeros(B, np.int32)
lib.permute(Z, fitted, resid, y, len(y), Z.shape[1], B, SEED, null, status)
assert np.isfinite(null).all()
extreme = int((abs(null) >= abs(r["beta"])).sum())
ci = stats.binomtest(extreme, B).proportion_ci()
r.update(
    residual_permutation_P=(extreme+1)/(B+1), permutations=B,
    permutation_extreme=extreme, MC95_low=ci.low, MC95_high=ci.high,
    permutation_nonconverged=int((status != 1).sum()), seed=SEED,
)
np.savez_compressed(OUT / "provenance_adjusted_permutation_null.npz", beta=null, status=status)

attenuation = 100 * (1 - r["beta"]/r0["beta"])
counts = data.groupby("pipeline").agg(
    n=("Subject","size"), events=("favourable6","sum"),
    FC_mean=(FEATURE,"mean"), FC_SD=(FEATURE,"std"),
).reset_index()
counts["non_events"] = counts.n - counts.events
counts.to_csv(OUT / "pipeline_group_composition.csv", index=False)
pd.DataFrame([
    dict(model="Primary clinical-adjusted model", **r0),
    dict(model="Primary model plus preprocessing provenance", **r),
]).to_csv(OUT / "model_comparison.csv", index=False)

coef = []
for k,name in enumerate([
    "Intercept", "baseline_CRS_R_sample_SD", "log1p_injury_to_stimulation_sample_SD",
    "fMRIPrep_vs_DPABI", "R_VPL_SMA_healthy_reference_SD"
]):
    coef.append(dict(term=name, **summary(fit, k)))
pd.DataFrame(coef).to_csv(OUT / "adjusted_model_all_coefficients.csv", index=False)

diagnostics = dict(
    design_rank=int(np.linalg.matrix_rank(X)), design_columns=X.shape[1],
    design_condition_number=float(np.linalg.cond(X)),
    max_accelerator_python_beta_difference=float(np.max(abs(cb-fit["beta"]))),
    max_accelerator_python_SE_difference=float(np.max(abs(cs-fit["se"]))),
    provenance_unknown_count=0, additional_patient_exclusions=0,
    coefficient_attenuation_percent=attenuation,
    upstream_provenance_is_not_scanner_site=True,
)

dump("RESULTS.json", dict(
    base=r0, provenance_adjusted=r, diagnostics=diagnostics,
    input_provenance={str(Path(q).name): sha(q) for q in [MASTER, LEDGER, BASELINE_RESULTS, ACCELERATOR]},
    software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform()),
))

report = f"""# Preprocessing-provenance sensitivity

Adding recorded upstream preprocessing provenance left the R_VPL-SMA association positive.
The odds ratio changed from {r0['OR']:.2f} to {r['OR']:.2f}
(95% Wald CI {r['CI95_low']:.2f}-{r['CI95_high']:.2f};
Wald P={r['Wald_P']:.4f}; 5,000-permutation P={r['residual_permutation_P']:.4f}).
The edge coefficient changed by {attenuation:.1f}%.

This analysis retains the primary 26-patient cohort, 12 events, outcome, healthy-reference
feature scale, baseline CRS-R and log1p injury-to-stimulation time. DPABI is coded 0 and
fMRIPrep 1. No patient is excluded because of the added covariate, and no feature is
reselected. The analysis tests robustness to recorded pipeline provenance; it does not
establish equivalence of the preprocessing workflows, remove all possible batch effects,
or provide independent patient validation.
"""
(OUT / "README_results.md").write_text(report, encoding="utf-8")
print(json.dumps(dict(base=r0, provenance_adjusted=r, diagnostics=diagnostics, output=str(OUT)), indent=2))

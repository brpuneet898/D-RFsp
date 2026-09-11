#!/usr/bin/env python3
"""
Conservative Estimated D-RFsp-I Monte Carlo Simulation
=======================================================

This script upgrades the original oracle-geometry proof-of-concept so that
D-RFsp-I ESTIMATES anisotropy from training data instead of receiving the true
(r, phi).

The simulation data-generating process remains aligned with the earlier script:
- 350 locations sampled uniformly on [0, 1]^2.
- Y = X1 + 0.5*X2 + W(s) + epsilon.
- W(s) has anisotropic exponential covariance.
- True principal direction phi = 30 degrees.
- True r scenarios: 1, 1.25, 1.5, 1.75, 2, 3, 5.
- Outer validation: top-right spatial block held out.
- RF, RFsp, and D-RFsp-I use the same Random Forest engine.
- RFsp is nested exactly at selected r = 1.

New estimation procedure, adapted from the supplied Conservative Upgraded
D-RFsp-I technical note:
1. Use OUTER-TRAINING data only.
2. Create cross-fitted residuals from a non-spatial baseline learner.
3. Use a directional residual variogram to obtain a training-only phi guide.
4. Build a local predictive (r, phi) candidate set around that guide.
5. Score candidates with four-fold INNER SPATIALLY BLOCKED CV.
6. Take the raw minimum-inner-RMSE candidate.
7. Conservative one-standard-error safeguard:
       choose r = 1 if
       isotropic_mean_RMSE <= raw_best_mean_RMSE + SE(raw_best fold RMSEs)
8. Refit the selected D-RFsp-I representation on all outer-training rows and
   evaluate only on the untouched outer test block.

Important simulation adaptation
-------------------------------
The Beijing note uses station-blocked folds because observations repeat over
12 monitoring stations. This synthetic experiment has one observation per
random spatial location, so the same leakage-safe principle is implemented
with four geographically contiguous spatial folds.

The fixed 5x5 reference grid from the original simulation is retained. Because
it is fixed before seeing any response values, it is leakage-safe.

Examples
--------
    python conservative_estimated_drfsp_simulation.py --simulations 1
    python conservative_estimated_drfsp_simulation.py --simulations 500 --jobs -1
    python conservative_estimated_drfsp_simulation.py --simulations 1000 --jobs -1

Optional selection diagnostics:
    python conservative_estimated_drfsp_simulation.py --simulations 100 --jobs -1 --diagnostics
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tabulate import tabulate


# ============================================================================
# SOURCE / EXPERIMENT-FIXED SETTINGS
# ============================================================================

N_LOCATIONS = 350
TRUE_PHI_DEGREES = 30.0
TRUE_R_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0)

SPATIAL_VARIANCE = 1.0
SPATIAL_RANGE = 0.25
NOISE_SD = 0.50
X1_SD = 1.0
X2_SD = 1.0

REFERENCE_GRID_SIZE = 5

# Outer held-out spatial block, unchanged from the first proof-of-concept.
BLOCK_X_CUTOFF = 0.50
BLOCK_Y_CUTOFF = 0.50

# Common final / screening Random Forest.
RF_N_ESTIMATORS = 20
RF_MAX_DEPTH = 18
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = 0.70

# Baseline residual learner from the technical note.
BASELINE_MAX_ITER = 150
BASELINE_LEARNING_RATE = 0.08
BASELINE_MAX_LEAF_NODES = 31
BASELINE_L2 = 1.0

# Structural variogram search from the technical note.
STRUCTURAL_PHI_VALUES = tuple(float(x) for x in range(0, 180, 5))
STRUCTURAL_R_VALUES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)

# Predictive r candidates from the technical note.
PREDICTIVE_R_VALUES = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
PREDICTIVE_PHI_OFFSETS = (-15.0, -5.0, 0.0, 5.0, 15.0)

INNER_FOLDS = 4

# Variogram fitting is structural guidance only. We fit the exponential range
# over a deterministic grid and solve sill/nugget coefficients by least squares.
VARIOGRAM_RANGE_GRID_SIZE = 25
VARIOGRAM_RANGE_MIN_FACTOR = 0.05
VARIOGRAM_RANGE_MAX_FACTOR = 2.00

DEFAULT_SEED = 20260909
CHOLESKY_JITTER = 1e-10


@dataclass(frozen=True)
class SimulationResult:
    """Results from one independently generated dataset at one true r."""

    repetition: int
    true_r: float
    rf_rmse: float
    rfsp_rmse: float
    drfsp_rmse: float

    # Stored for optional diagnostics; primary result table remains unchanged.
    phi_guide: float
    raw_r: float
    raw_phi: float
    selected_r: float
    selected_phi: float
    raw_best_cv: float
    isotropic_cv: float
    winner_se: float


# ============================================================================
# DISTANCE GEOMETRY
# ============================================================================

def make_reference_points(grid_size: int = REFERENCE_GRID_SIZE) -> np.ndarray:
    """Create the fixed regular reference grid on [0, 1]^2."""
    axis = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def euclidean_distances(points: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Ordinary RFsp Euclidean distance-to-reference predictors."""
    delta = points[:, None, :] - refs[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def anisotropic_distances(
    points: np.ndarray,
    refs: np.ndarray,
    phi_degrees: float,
    r: float,
) -> np.ndarray:
    """
    Directional distance:
        d_A = sqrt(u^2 + r^2 v^2)

    At r=1 this function deliberately returns the exact same Euclidean matrix
    used by RFsp.
    """
    if np.isclose(r, 1.0):
        return euclidean_distances(points, refs)

    phi = math.radians(phi_degrees % 180.0)
    delta = points[:, None, :] - refs[None, :, :]
    dx = delta[:, :, 0]
    dy = delta[:, :, 1]

    c = math.cos(phi)
    s = math.sin(phi)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return np.sqrt(u * u + (r * r) * v * v)


def pairwise_anisotropic_distance_from_deltas(
    dx: np.ndarray,
    dy: np.ndarray,
    phi_degrees: float,
    r: float,
) -> np.ndarray:
    """Vectorized anisotropic distances for precomputed pair displacements."""
    if np.isclose(r, 1.0):
        return np.sqrt(dx * dx + dy * dy)

    phi = math.radians(phi_degrees % 180.0)
    c = math.cos(phi)
    s = math.sin(phi)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return np.sqrt(u * u + (r * r) * v * v)


# ============================================================================
# DATA GENERATION
# ============================================================================

def spatial_covariance(
    points: np.ndarray,
    phi_degrees: float,
    r: float,
    variance: float = SPATIAL_VARIANCE,
    spatial_range: float = SPATIAL_RANGE,
) -> np.ndarray:
    """Anisotropic exponential covariance sigma^2 exp(-d_A/range)."""
    d = anisotropic_distances(points, points, phi_degrees, r)
    cov = variance * np.exp(-d / spatial_range)
    cov.flat[:: cov.shape[0] + 1] += CHOLESKY_JITTER
    return cov


def generate_dataset(
    rng: np.random.Generator,
    true_r: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one n=350 synthetic dataset for a specified true anisotropy r."""
    locations = rng.uniform(0.0, 1.0, size=(N_LOCATIONS, 2))
    x1 = rng.normal(0.0, X1_SD, size=N_LOCATIONS)
    x2 = rng.normal(0.0, X2_SD, size=N_LOCATIONS)

    cov = spatial_covariance(locations, TRUE_PHI_DEGREES, true_r)
    spatial_effect = np.linalg.cholesky(cov) @ rng.normal(size=N_LOCATIONS)
    epsilon = rng.normal(0.0, NOISE_SD, size=N_LOCATIONS)

    y = x1 + 0.5 * x2 + spatial_effect + epsilon
    x_nonspatial = np.column_stack([x1, x2])
    return locations, x_nonspatial, y


# ============================================================================
# OUTER AND INNER SPATIAL BLOCKS
# ============================================================================

def outer_spatial_block_split(locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out the top-right spatial quadrant exactly as in the original script."""
    test_mask = (
        (locations[:, 0] >= BLOCK_X_CUTOFF)
        & (locations[:, 1] >= BLOCK_Y_CUTOFF)
    )
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)

    if len(train_idx) < 40 or len(test_idx) < 20:
        raise RuntimeError(
            f"Outer spatial split produced too few observations: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )
    return train_idx, test_idx


def make_four_spatial_folds(locations: np.ndarray) -> np.ndarray:
    """
    Assign locations to four geographically contiguous folds.

    Median x and y cut the CURRENT OUTER-TRAINING region into four blocks.
    If an extremely unusual random sample produces an empty block, deterministic
    coordinate ordering is used as a safe fallback.
    """
    x = locations[:, 0]
    y = locations[:, 1]
    x_med = float(np.median(x))
    y_med = float(np.median(y))

    fold = (x >= x_med).astype(int) + 2 * (y >= y_med).astype(int)
    counts = np.bincount(fold, minlength=4)

    if np.all(counts >= 2):
        return fold

    # Rare fallback: preserve deterministic spatial ordering.
    order = np.lexsort((y, x))
    fallback = np.empty(len(locations), dtype=int)
    chunks = np.array_split(order, 4)
    for k, idx in enumerate(chunks):
        fallback[idx] = k
    return fallback


# ============================================================================
# COMMON LEARNERS
# ============================================================================

def make_rf(seed: int) -> RandomForestRegressor:
    """Same RF engine for RF, RFsp, and every D-RFsp-I candidate."""
    return RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        random_state=seed,
        n_jobs=1,  # outer Monte Carlo parallelization is handled by joblib
    )


def make_baseline(seed: int) -> HistGradientBoostingRegressor:
    """Non-spatial learner for cross-fitted residual construction."""
    return HistGradientBoostingRegressor(
        max_iter=BASELINE_MAX_ITER,
        learning_rate=BASELINE_LEARNING_RATE,
        max_leaf_nodes=BASELINE_MAX_LEAF_NODES,
        l2_regularization=BASELINE_L2,
        random_state=seed,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================================
# CROSS-FITTED RESIDUALS
# ============================================================================

def cross_fitted_residuals(
    x_nonspatial: np.ndarray,
    y: np.ndarray,
    inner_fold_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Build residuals using predictions from baseline models that never train on
    the observation's spatial validation fold.
    """
    residuals = np.empty_like(y, dtype=float)

    for k in range(INNER_FOLDS):
        val = inner_fold_ids == k
        train = ~val

        model = make_baseline(seed + 100 + k)
        model.fit(x_nonspatial[train], y[train])
        residuals[val] = y[val] - model.predict(x_nonspatial[val])

    return residuals


# ============================================================================
# DIRECTIONAL VARIOGRAM GUIDE
# ============================================================================

def pairwise_variogram_inputs(
    locations: np.ndarray,
    residuals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upper-triangle dx, dy, and empirical residual semivariance."""
    i, j = np.triu_indices(len(locations), k=1)
    dx = locations[i, 0] - locations[j, 0]
    dy = locations[i, 1] - locations[j, 1]
    gamma = 0.5 * (residuals[i] - residuals[j]) ** 2
    return dx, dy, gamma


def exponential_variogram_sse(distances: np.ndarray, gamma: np.ndarray) -> float:
    """
    Fit gamma(d)=c0+c1*(1-exp(-d/a)) approximately.

    For a deterministic grid of range a, solve c0/c1 by least squares and keep
    the smallest SSE. Nonnegative c0 and c1 are enforced by clipping and then
    recomputing fitted values. This stage only ranks structural candidates.
    """
    positive = distances[distances > 0]
    if len(positive) == 0:
        return float("inf")

    scale = float(np.median(positive))
    a_grid = np.geomspace(
        max(scale * VARIOGRAM_RANGE_MIN_FACTOR, 1e-6),
        max(scale * VARIOGRAM_RANGE_MAX_FACTOR, 2e-6),
        VARIOGRAM_RANGE_GRID_SIZE,
    )

    best_sse = float("inf")
    ones = np.ones_like(distances)

    for a in a_grid:
        basis = 1.0 - np.exp(-distances / a)
        design = np.column_stack([ones, basis])
        coef, *_ = np.linalg.lstsq(design, gamma, rcond=None)
        c0 = max(float(coef[0]), 0.0)
        c1 = max(float(coef[1]), 0.0)
        fitted = c0 + c1 * basis
        sse = float(np.sum((gamma - fitted) ** 2))
        if sse < best_sse:
            best_sse = sse

    return best_sse


def directional_variogram_guide(
    locations: np.ndarray,
    residuals: np.ndarray,
) -> float:
    """
    Search the structural (phi, r) grid using OUTER-TRAINING residuals only.

    If the overall best structural candidate is isotropic, use the best
    non-isotropic candidate only to identify phi, while isotropy remains fully
    available in predictive selection.
    """
    dx, dy, gamma = pairwise_variogram_inputs(locations, residuals)

    best_all = (float("inf"), 0.0, 1.0)
    best_noniso = (float("inf"), 0.0, 1.25)

    # r=1 is independent of phi, so evaluate it once.
    d_iso = pairwise_anisotropic_distance_from_deltas(dx, dy, 0.0, 1.0)
    iso_sse = exponential_variogram_sse(d_iso, gamma)
    best_all = (iso_sse, 0.0, 1.0)

    for r in STRUCTURAL_R_VALUES:
        if np.isclose(r, 1.0):
            continue
        for phi in STRUCTURAL_PHI_VALUES:
            d = pairwise_anisotropic_distance_from_deltas(dx, dy, phi, r)
            sse = exponential_variogram_sse(d, gamma)

            if sse < best_noniso[0]:
                best_noniso = (sse, phi, r)
            if sse < best_all[0]:
                best_all = (sse, phi, r)

    if np.isclose(best_all[2], 1.0):
        return float(best_noniso[1])
    return float(best_all[1])


def predictive_candidates(phi_guide: float) -> List[Tuple[float, float]]:
    """
    Build the 26-candidate predictive set:
      r=1 once
      5 non-isotropic r values x 5 local angles
    """
    candidates: List[Tuple[float, float]] = [(1.0, float("nan"))]

    local_phis = []
    for offset in PREDICTIVE_PHI_OFFSETS:
        phi = (phi_guide + offset) % 180.0
        if not any(np.isclose(phi, p) for p in local_phis):
            local_phis.append(phi)

    for r in PREDICTIVE_R_VALUES:
        if np.isclose(r, 1.0):
            continue
        for phi in local_phis:
            candidates.append((float(r), float(phi)))

    return candidates


# ============================================================================
# INNER BLOCKED PREDICTIVE SELECTION
# ============================================================================

def candidate_features(
    locations: np.ndarray,
    x_nonspatial: np.ndarray,
    refs: np.ndarray,
    r: float,
    phi: float,
) -> np.ndarray:
    """Build RFsp / D-RFsp-I features for one candidate geometry."""
    if np.isclose(r, 1.0):
        d = euclidean_distances(locations, refs)
    else:
        d = anisotropic_distances(locations, refs, phi, r)
    return np.column_stack([x_nonspatial, d])


def score_candidate_inner_cv(
    features: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    base_seed: int,
    candidate_index: int,
) -> np.ndarray:
    """Return foldwise blocked-CV RMSE values for one candidate."""
    fold_scores = []

    for k in range(INNER_FOLDS):
        val = fold_ids == k
        train = ~val

        model_seed = base_seed + 10_000 + candidate_index * 100 + k
        model = make_rf(model_seed)
        model.fit(features[train], y[train])
        pred = model.predict(features[val])
        fold_scores.append(rmse(y[val], pred))

    return np.asarray(fold_scores, dtype=float)


def estimate_geometry_conservatively(
    locations: np.ndarray,
    x_nonspatial: np.ndarray,
    y: np.ndarray,
    refs: np.ndarray,
    seed: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Estimate phi/r completely inside outer training data.

    Returns
    -------
    phi_guide,
    raw_r, raw_phi,
    selected_r, selected_phi,
    raw_best_mean, isotropic_mean, winner_se
    """
    fold_ids = make_four_spatial_folds(locations)

    residuals = cross_fitted_residuals(
        x_nonspatial=x_nonspatial,
        y=y,
        inner_fold_ids=fold_ids,
        seed=seed,
    )

    phi_guide = directional_variogram_guide(locations, residuals)
    candidates = predictive_candidates(phi_guide)

    records = []
    for idx, (r, phi) in enumerate(candidates):
        features = candidate_features(
            locations=locations,
            x_nonspatial=x_nonspatial,
            refs=refs,
            r=r,
            phi=phi,
        )
        fold_scores = score_candidate_inner_cv(
            features=features,
            y=y,
            fold_ids=fold_ids,
            base_seed=seed,
            candidate_index=idx,
        )
        records.append(
            {
                "r": r,
                "phi": phi,
                "fold_scores": fold_scores,
                "mean": float(np.mean(fold_scores)),
            }
        )

    raw = min(records, key=lambda z: z["mean"])
    iso = next(z for z in records if np.isclose(z["r"], 1.0))

    raw_scores = raw["fold_scores"]
    winner_se = float(np.std(raw_scores, ddof=1) / np.sqrt(len(raw_scores)))

    if iso["mean"] <= raw["mean"] + winner_se:
        selected_r = 1.0
        selected_phi = float("nan")
    else:
        selected_r = float(raw["r"])
        selected_phi = float(raw["phi"])

    return (
        float(phi_guide),
        float(raw["r"]),
        float(raw["phi"]),
        selected_r,
        selected_phi,
        float(raw["mean"]),
        float(iso["mean"]),
        winner_se,
    )


# ============================================================================
# FINAL OUTER-FOLD MODEL FITS
# ============================================================================

def fit_predict_rmse(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> float:
    model = make_rf(seed)
    model.fit(features[train_idx], y[train_idx])
    return rmse(y[test_idx], model.predict(features[test_idx]))


def run_one_simulation(
    repetition: int,
    true_r: float,
    master_seed: int,
) -> SimulationResult:
    """Run one complete leakage-safe estimated-geometry simulation."""
    r_code = int(round(true_r * 1000))
    seq = np.random.SeedSequence([master_seed, repetition, r_code])
    data_seq, model_seq = seq.spawn(2)

    rng = np.random.default_rng(data_seq)
    base_model_seed = int(model_seq.generate_state(1, dtype=np.uint32)[0])

    locations, x_nonspatial, y = generate_dataset(rng, true_r)
    train_idx, test_idx = outer_spatial_block_split(locations)
    refs = make_reference_points()

    # Common full-data feature matrices.
    rf_features = x_nonspatial
    rfsp_features = np.column_stack(
        [x_nonspatial, euclidean_distances(locations, refs)]
    )

    # RF and RFsp outer evaluation.
    rf_rmse = fit_predict_rmse(
        rf_features, y, train_idx, test_idx, base_model_seed + 1
    )
    rfsp_rmse = fit_predict_rmse(
        rfsp_features, y, train_idx, test_idx, base_model_seed + 2
    )

    # Estimate D-RFsp-I geometry using OUTER TRAINING ONLY.
    (
        phi_guide,
        raw_r,
        raw_phi,
        selected_r,
        selected_phi,
        raw_best_cv,
        isotropic_cv,
        winner_se,
    ) = estimate_geometry_conservatively(
        locations=locations[train_idx],
        x_nonspatial=x_nonspatial[train_idx],
        y=y[train_idx],
        refs=refs,
        seed=base_model_seed + 50_000,
    )

    # Exact nesting: selected r=1 means exactly the RFsp representation.
    if np.isclose(selected_r, 1.0):
        drfsp_rmse = rfsp_rmse
    else:
        drfsp_features = candidate_features(
            locations=locations,
            x_nonspatial=x_nonspatial,
            refs=refs,
            r=selected_r,
            phi=selected_phi,
        )
        drfsp_rmse = fit_predict_rmse(
            drfsp_features, y, train_idx, test_idx, base_model_seed + 3
        )

    return SimulationResult(
        repetition=repetition,
        true_r=true_r,
        rf_rmse=rf_rmse,
        rfsp_rmse=rfsp_rmse,
        drfsp_rmse=drfsp_rmse,
        phi_guide=phi_guide,
        raw_r=raw_r,
        raw_phi=raw_phi,
        selected_r=selected_r,
        selected_phi=selected_phi,
        raw_best_cv=raw_best_cv,
        isotropic_cv=isotropic_cv,
        winner_se=winner_se,
    )


# ============================================================================
# MONTE CARLO DRIVER AND OUTPUT
# ============================================================================

def run_monte_carlo(
    n_simulations: int,
    true_r_values: Iterable[float],
    master_seed: int,
    jobs: int,
) -> List[SimulationResult]:
    tasks = [
        (rep, true_r)
        for true_r in true_r_values
        for rep in range(1, n_simulations + 1)
    ]

    if jobs == 1:
        return [
            run_one_simulation(rep, true_r, master_seed)
            for rep, true_r in tasks
        ]

    return Parallel(
        n_jobs=jobs,
        verbose=0,
        batch_size="auto",
    )(
        delayed(run_one_simulation)(rep, true_r, master_seed)
        for rep, true_r in tasks
    )


def summarize_results(
    results: Sequence[SimulationResult],
) -> List[List[float]]:
    """Primary table intentionally matches the original script's format."""
    rows: List[List[float]] = []

    for true_r in TRUE_R_VALUES:
        subset = [x for x in results if np.isclose(x.true_r, true_r)]

        rf = np.asarray([x.rf_rmse for x in subset], dtype=float)
        rfsp = np.asarray([x.rfsp_rmse for x in subset], dtype=float)
        drfsp = np.asarray([x.drfsp_rmse for x in subset], dtype=float)

        gain_pct = 100.0 * (rfsp.mean() - drfsp.mean()) / rfsp.mean()

        rows.append(
            [
                true_r,
                rf.mean(),
                rfsp.mean(),
                drfsp.mean(),
                gain_pct,
            ]
        )

    return rows


def summarize_selection_diagnostics(
    results: Sequence[SimulationResult],
) -> List[List[float]]:
    """Optional parameter-selection summary; not shown unless requested."""
    rows = []

    for true_r in TRUE_R_VALUES:
        subset = [x for x in results if np.isclose(x.true_r, true_r)]
        selected_r = np.asarray([x.selected_r for x in subset], dtype=float)
        raw_r = np.asarray([x.raw_r for x in subset], dtype=float)

        anisotropic_pct = 100.0 * np.mean(selected_r > 1.0 + 1e-12)
        iso_pct = 100.0 - anisotropic_pct
        mean_selected_r = float(np.mean(selected_r))
        mean_raw_r = float(np.mean(raw_r))

        # Axial angular error only where final anisotropy survived.
        errors = []
        for x in subset:
            if x.selected_r > 1.0 and np.isfinite(x.selected_phi):
                d = abs((x.selected_phi - TRUE_PHI_DEGREES) % 180.0)
                errors.append(min(d, 180.0 - d))
        mean_phi_error = float(np.mean(errors)) if errors else float("nan")

        rows.append(
            [
                true_r,
                iso_pct,
                anisotropic_pct,
                mean_raw_r,
                mean_selected_r,
                mean_phi_error,
            ]
        )

    return rows


def print_settings(
    n_simulations: int,
    jobs: int,
    seed: int,
) -> None:
    print("\nConservative Estimated D-RFsp-I Monte Carlo Simulation")
    print("=" * 78)
    print(f"Simulations per true r  : {n_simulations}")
    print(f"True r values           : {list(TRUE_R_VALUES)}")
    print(f"True phi                : {TRUE_PHI_DEGREES:.1f} degrees")
    print(f"Locations per dataset   : {N_LOCATIONS}")
    print(f"Spatial variance        : {SPATIAL_VARIANCE}")
    print(f"Spatial range           : {SPATIAL_RANGE}")
    print(f"Noise SD                : {NOISE_SD}")
    print(
        f"Reference points        : {REFERENCE_GRID_SIZE}x{REFERENCE_GRID_SIZE} "
        f"grid ({REFERENCE_GRID_SIZE ** 2})"
    )
    print("Outer validation block  : x >= 0.50 AND y >= 0.50")
    print(f"Inner spatial folds     : {INNER_FOLDS}")
    print(f"Structural phi grid     : 0, 5, ..., 175 degrees")
    print(f"Structural r grid       : {list(STRUCTURAL_R_VALUES)}")
    print(f"Predictive r candidates : {list(PREDICTIVE_R_VALUES)}")
    print(f"Predictive phi offsets  : {list(PREDICTIVE_PHI_OFFSETS)} degrees")
    print("Conservative safeguard  : isotropic if L_iso <= L_best + SE_best")
    print(
        "RF settings             : "
        f"trees={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}, "
        f"min_leaf={RF_MIN_SAMPLES_LEAF}, max_features={RF_MAX_FEATURES}"
    )
    print(f"Master seed             : {seed}")
    print(f"Parallel jobs           : {jobs}")
    print("=" * 78)


def print_primary_summary(
    rows: List[List[float]],
    elapsed_seconds: float,
) -> None:
    headers = [
        "True r",
        "RF avg RMSE",
        "RFsp avg RMSE",
        "D-RFsp avg RMSE",
        "D-RFsp gain vs RFsp (%)",
    ]

    print("\nAVERAGED BLOCKED-TEST RMSE RESULTS")
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            floatfmt=(".2f", ".5f", ".5f", ".5f", ".3f"),
        )
    )
    print(f"\nElapsed time: {elapsed_seconds:.2f} seconds")
    print(
        "Interpretation: positive gain (%) means lower average RMSE for the "
        "conservatively estimated D-RFsp-I than RFsp."
    )
    print(
        "Important: 'True r' is used only to generate the synthetic field. "
        "D-RFsp-I does NOT receive the true r or true phi."
    )


def print_diagnostics(rows: List[List[float]]) -> None:
    headers = [
        "True r",
        "Final r=1 (%)",
        "Final r>1 (%)",
        "Mean raw r",
        "Mean selected r",
        "Mean axial phi error (deg)",
    ]

    print("\nOPTIONAL SELECTION DIAGNOSTICS")
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            floatfmt=(".2f", ".2f", ".2f", ".3f", ".3f", ".2f"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe Monte Carlo simulation for RF, RFsp, and "
            "conservative estimated D-RFsp-I."
        )
    )
    parser.add_argument(
        "--simulations",
        "-n",
        type=int,
        default=1,
        help=(
            "Independent simulations PER true-r scenario "
            "(e.g. 1, 500, 1000). Default: 1"
        ),
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help=(
            "Parallel workers across independent simulations. "
            "Use -1 for all CPU cores. Default: 1"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Master random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Also print r-selection frequency and phi-error diagnostics. "
            "Primary RMSE table remains unchanged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.simulations < 1:
        raise ValueError("--simulations must be >= 1")
    if args.jobs == 0:
        raise ValueError("--jobs cannot be 0; use 1, a positive integer, or -1")

    print_settings(args.simulations, args.jobs, args.seed)

    start = time.perf_counter()
    results = run_monte_carlo(
        n_simulations=args.simulations,
        true_r_values=TRUE_R_VALUES,
        master_seed=args.seed,
        jobs=args.jobs,
    )
    elapsed = time.perf_counter() - start

    primary_rows = summarize_results(results)
    print_primary_summary(primary_rows, elapsed)

    if args.diagnostics:
        diagnostic_rows = summarize_selection_diagnostics(results)
        print_diagnostics(diagnostic_rows)


if __name__ == "__main__":
    main()

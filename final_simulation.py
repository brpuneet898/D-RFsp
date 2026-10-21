#!/usr/bin/env python3
"""
final_simulation.py
===================

Final two-stage manuscript simulation for:

    Directional Random Forest for Spatial Prediction under Geometric Anisotropy

The script implements the frozen simulation roadmap:

Simulation I — parameter recovery
    * estimate r and phi from TRAINING DATA ONLY in every replication;
    * summarize mean r_hat, SD, bias, RMSE(r_hat);
    * summarize axial angular MAE and RMSE(phi_hat).

Simulation II — predictive comparison
    * RF
    * RFsp
    * estimated D-RFsp (PZ-REML geometry estimated from training data only)
    * optional oracle D-RFsp using the true (r0, phi0)
    * summarize mean RMSE, SD(RMSE), mean MAE, mean R^2,
      and Delta RMSE versus RFsp.

Primary manuscript design
    B = 1000 replications
    r0 in {1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5}
    phi0 in {0, 30, 60} degrees (axial angles)
    n in {350, 1000} (moderate, large)
    nugget/noise SD in {0.25, 0.50, 1.00} (low, moderate, high)

Data-generating mechanism
    locations ~ Uniform([0,1]^2)
    X1, X2 ~ N(0,1)
    Y = X1 + 0.5*X2 + W(s) + epsilon
    W(s): zero-mean GP with exponential covariance under
          determinant-normalized geometric anisotropy
    epsilon ~ N(0, noise_sd^2)

Frozen geometry
    kappa = log(r)
    z1 = kappa*cos(2*phi)
    z2 = kappa*sin(2*phi)

    H(kappa,phi) = R(phi)^T diag(exp(-kappa), exp(kappa)) R(phi)
    det(H) = 1

    d_H^2 = u^2/r + r*v^2

where u is separation along the principal axis and v is cross-axis
separation.  At r=1, d_H is exactly Euclidean distance, so D-RFsp
nests RFsp.

PZ-REML
    Frozen pipeline:
        isotropic REML start
        -> geometry-informed Z start
        -> joint determinant-normalized Z-REML
        -> fixed mild radial penalty toward isotropy
        -> recover (r_hat, phi_hat)

    Penalty:
        lambda_kappa * ||Z||_2
    with
        lambda_kappa = -log(0.30)/log(3)

    The covariance fit is capped at a spatially balanced subset of
    training observations (default 100), matching the project's frozen
    practical implementation for tractable repeated REML estimation.

Output
    The requested manuscript-ready results are written incrementally to:

        final_simulation_results.md

    The file contains:
        * full configuration;
        * Simulation I parameter-recovery table;
        * Simulation II predictive-comparison table;
        * scenario-completion/progress section;
        * interpretation notes.

Examples
--------
Full manuscript run (VERY computationally intensive):
    python final_simulation.py

Parallel full run:
    python final_simulation.py --jobs -1

Quick smoke test:
    python final_simulation.py --smoke-test

Short validation run:
    python final_simulation.py --replications 10 --n-values 350 \
        --r-values 1 2 3 --phi-values 0 30 60 --noise-values 0.5

Resume:
    python final_simulation.py --resume

Notes
-----
1. The full B=1000 factorial design is intentionally large. Run on a
   multi-core workstation/HPC environment if available.
2. All geometry estimation is performed strictly from the training split.
   Held-out outcomes are never used for anisotropy estimation or tuning.
3. phi is axial: phi and phi+180 degrees are equivalent.
4. When r0=1, orientation is not identifiable. Phi-error summaries are
   therefore reported as NA for the isotropic scenario.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Frozen manuscript defaults
# ---------------------------------------------------------------------

DEFAULT_B = 1000
DEFAULT_R_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0)
DEFAULT_PHI_VALUES = (0.0, 30.0, 60.0)      # degrees; axial
DEFAULT_N_VALUES = (350, 1000)              # moderate, large
DEFAULT_NOISE_VALUES = (0.25, 0.50, 1.00)   # low, moderate, high

SPATIAL_VARIANCE = 1.0
SPATIAL_RANGE = 0.25
X1_SD = 1.0
X2_SD = 1.0

REFERENCE_GRID_SIZE = 5

# Same blocked philosophy as the earlier simulation: top-right block is held out.
BLOCK_X_CUTOFF = 0.50
BLOCK_Y_CUTOFF = 0.50

# Same RF engine is used for RFsp and both D-RFsp variants.
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = 0.70

# PZ-REML settings
PZ_LAMBDA_KAPPA = -math.log(0.30) / math.log(3.0)
PZ_R_MAX = 10.0
PZ_ESTIMATION_MAX_N = 100
PZ_RANGE_BOUNDS = (0.02, 2.0)
PZ_NUGGET_RATIO_BOUNDS = (1e-4, 10.0)
PZ_MAXITER = 120
PZ_FTOL = 1e-8

DEFAULT_SEED = 20260909
CHOLESKY_JITTER = 1e-9
RESULTS_FILE = "final_simulation_results.md"
CHECKPOINT_FILE = ".final_simulation_checkpoint.jsonl"


# ---------------------------------------------------------------------
# Utility structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    n: int
    r0: float
    phi0_deg: float
    noise_sd: float

    @property
    def key(self) -> str:
        return f"n={self.n}|r={self.r0:g}|phi={self.phi0_deg:g}|noise={self.noise_sd:g}"


@dataclass
class ReplicationResult:
    scenario_key: str
    repetition: int
    seed: int

    r_hat: float
    phi_hat_deg: float
    kappa_hat: float
    reml_success: bool
    reml_objective: float

    rf_rmse: float
    rf_mae: float
    rf_r2: float

    rfsp_rmse: float
    rfsp_mae: float
    rfsp_r2: float

    drfsp_rmse: float
    drfsp_mae: float
    drfsp_r2: float

    oracle_rmse: float
    oracle_mae: float
    oracle_r2: float


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def normalize_axial_deg(phi_deg: float) -> float:
    """Map an axial angle to [0, 180)."""
    return float(phi_deg % 180.0)


def axial_error_deg(phi_hat_deg: float, phi0_deg: float) -> float:
    """
    Smallest absolute axial angular error in degrees, in [0, 90].
    phi and phi+180 are equivalent.
    """
    a = normalize_axial_deg(phi_hat_deg)
    b = normalize_axial_deg(phi0_deg)
    d = abs(a - b)
    return float(min(d, 180.0 - d))


def z_from_r_phi(r: float, phi_deg: float) -> Tuple[float, float]:
    kappa = max(0.0, math.log(max(float(r), 1.0)))
    phi = math.radians(normalize_axial_deg(phi_deg))
    return (
        kappa * math.cos(2.0 * phi),
        kappa * math.sin(2.0 * phi),
    )


def r_phi_from_z(z1: float, z2: float) -> Tuple[float, float, float]:
    """
    Recover r, phi, kappa from smooth axial coordinates Z=(z1,z2).
    phi is returned in [0,180) degrees.
    """
    kappa = float(math.hypot(z1, z2))
    r = float(math.exp(kappa))
    if kappa < 1e-12:
        phi_deg = 0.0
    else:
        phi = 0.5 * math.atan2(z2, z1)
        phi_deg = normalize_axial_deg(math.degrees(phi))
    return r, phi_deg, kappa


def determinant_normalized_distances(
    points_a: np.ndarray,
    points_b: np.ndarray,
    r: float,
    phi_deg: float,
) -> np.ndarray:
    """
    Pairwise determinant-normalized geometric-anisotropy distance.

        H = R^T diag(1/r, r) R
        d_H^2 = u^2/r + r*v^2
        det(H)=1

    r=1 gives exact Euclidean distance.
    """
    r = max(float(r), 1.0)
    delta = points_a[:, None, :] - points_b[None, :, :]
    dx = delta[..., 0]
    dy = delta[..., 1]

    if abs(r - 1.0) < 1e-14:
        return np.sqrt(dx * dx + dy * dy)

    phi = math.radians(normalize_axial_deg(phi_deg))
    c, s = math.cos(phi), math.sin(phi)
    u = dx * c + dy * s
    v = -dx * s + dy * c

    d2 = (u * u) / r + r * (v * v)
    return np.sqrt(np.maximum(d2, 0.0))


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def spatial_covariance(
    locations: np.ndarray,
    r: float,
    phi_deg: float,
    spatial_range: float = SPATIAL_RANGE,
    variance: float = SPATIAL_VARIANCE,
) -> np.ndarray:
    d = determinant_normalized_distances(locations, locations, r, phi_deg)
    cov = variance * np.exp(-d / spatial_range)
    cov.flat[:: cov.shape[0] + 1] += CHOLESKY_JITTER
    return cov


def generate_dataset(
    rng: np.random.Generator,
    n: int,
    r0: float,
    phi0_deg: float,
    noise_sd: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    locations = rng.uniform(0.0, 1.0, size=(n, 2))
    x1 = rng.normal(0.0, X1_SD, size=n)
    x2 = rng.normal(0.0, X2_SD, size=n)

    cov = spatial_covariance(locations, r0, phi0_deg)
    L = np.linalg.cholesky(cov)
    w = L @ rng.normal(size=n)

    eps = rng.normal(0.0, noise_sd, size=n)
    y = x1 + 0.5 * x2 + w + eps

    x_nonspatial = np.column_stack([x1, x2])
    return locations, x_nonspatial, y


def spatial_block_split(locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed blocked split: hold out the top-right spatial quadrant.
    """
    test_mask = (
        (locations[:, 0] >= BLOCK_X_CUTOFF)
        & (locations[:, 1] >= BLOCK_Y_CUTOFF)
    )
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)

    if len(train_idx) < 30 or len(test_idx) < 20:
        raise RuntimeError(
            f"Blocked split too small: train={len(train_idx)}, test={len(test_idx)}"
        )
    return train_idx, test_idx


# ---------------------------------------------------------------------
# Balanced covariance-estimation subset
# ---------------------------------------------------------------------

def balanced_subset_indices(
    locations: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
    grid_side: int = 5,
) -> np.ndarray:
    """
    Spatially balanced subset for repeated REML estimation.

    The unit square is divided into grid cells and observations are sampled
    approximately evenly across occupied cells. This uses only locations.
    """
    n = len(locations)
    if n <= max_n:
        return np.arange(n, dtype=int)

    xbin = np.minimum((locations[:, 0] * grid_side).astype(int), grid_side - 1)
    ybin = np.minimum((locations[:, 1] * grid_side).astype(int), grid_side - 1)
    cell = ybin * grid_side + xbin

    occupied = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    occupied = [idx.copy() for idx in occupied if len(idx) > 0]
    for idx in occupied:
        rng.shuffle(idx)

    selected: List[int] = []
    ptr = [0] * len(occupied)

    while len(selected) < max_n:
        added = False
        for j, idx in enumerate(occupied):
            if len(selected) >= max_n:
                break
            if ptr[j] < len(idx):
                selected.append(int(idx[ptr[j]]))
                ptr[j] += 1
                added = True
        if not added:
            break

    return np.asarray(selected, dtype=int)


# ---------------------------------------------------------------------
# PZ-REML estimator
# ---------------------------------------------------------------------

def _profile_reml_nll(
    theta: np.ndarray,
    locations: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    penalty_lambda: float,
    r_max: float,
) -> float:
    """
    Profiled Gaussian REML objective.

    theta = [log_range, log_nugget_ratio, z1, z2]

    Correlation model:
        K = exp(-d_H / range) + nugget_ratio * I

    sigma^2 is profiled out. The fixed-effect design X is included in REML.
    """
    log_range, log_eta, z1, z2 = map(float, theta)
    spatial_range = math.exp(log_range)
    eta = math.exp(log_eta)

    r, phi_deg, kappa = r_phi_from_z(z1, z2)
    if not np.isfinite(r) or r > r_max * (1.0 + 1e-10):
        excess = max(0.0, kappa - math.log(r_max))
        return 1e10 + 1e8 * excess * excess

    try:
        d = determinant_normalized_distances(locations, locations, r, phi_deg)
        K = np.exp(-d / spatial_range)
        K.flat[:: K.shape[0] + 1] += eta + CHOLESKY_JITTER

        cK, lower = cho_factor(K, lower=True, check_finite=False)
        Ki_y = cho_solve((cK, lower), y, check_finite=False)
        Ki_X = cho_solve((cK, lower), X, check_finite=False)

        XtKiX = X.T @ Ki_X
        cX, lowerX = cho_factor(XtKiX, lower=True, check_finite=False)
        beta_hat = cho_solve((cX, lowerX), X.T @ Ki_y, check_finite=False)

        resid = y - X @ beta_hat
        Ki_resid = cho_solve((cK, lower), resid, check_finite=False)
        q = float(resid @ Ki_resid)

        n, p = X.shape
        df = n - p
        if q <= 0.0 or df <= 0:
            return 1e12

        logdetK = 2.0 * np.sum(np.log(np.diag(cK)))
        logdetXtKiX = 2.0 * np.sum(np.log(np.diag(cX)))

        # Constants common to all parameter values are omitted.
        reml = logdetK + logdetXtKiX + df * math.log(q / df)

        # Frozen mild PZ penalty toward isotropy.
        penalty = penalty_lambda * kappa

        val = float(reml + penalty)
        if not np.isfinite(val):
            return 1e12
        return val

    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return 1e12


def _isotropic_start(
    locations: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Stage 1: isotropic REML start, optimizing range and nugget ratio only.
    """
    x0 = np.array([math.log(0.25), math.log(0.25), 0.0, 0.0], dtype=float)

    def obj2(v: np.ndarray) -> float:
        theta = np.array([v[0], v[1], 0.0, 0.0], dtype=float)
        return _profile_reml_nll(
            theta, locations, y, X,
            penalty_lambda=0.0,
            r_max=PZ_R_MAX,
        )

    b = [
        (math.log(PZ_RANGE_BOUNDS[0]), math.log(PZ_RANGE_BOUNDS[1])),
        (math.log(PZ_NUGGET_RATIO_BOUNDS[0]), math.log(PZ_NUGGET_RATIO_BOUNDS[1])),
    ]
    opt = minimize(
        obj2,
        x0[:2],
        method="L-BFGS-B",
        bounds=b,
        options={"maxiter": 80, "ftol": 1e-8},
    )
    if opt.success and np.all(np.isfinite(opt.x)):
        x0[0:2] = opt.x
    return x0


def _geometry_informed_z_start(
    locations: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
) -> Tuple[float, float]:
    """
    Stage 2: inexpensive geometry-informed start.

    Residualize y against X, inspect short-to-medium separation pairs,
    and choose the axial direction with the smallest directional
    semivariance. This is used ONLY as an optimizer start; the final
    estimate is the joint PZ-REML solution.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    n = len(y)
    ii, jj = np.triu_indices(n, k=1)
    dx = locations[jj, 0] - locations[ii, 0]
    dy = locations[jj, 1] - locations[ii, 1]
    dist = np.hypot(dx, dy)

    if len(dist) < 20:
        return 0.0, 0.0

    positive = dist > 1e-10
    if not np.any(positive):
        return 0.0, 0.0

    cutoff = np.quantile(dist[positive], 0.50)
    keep = positive & (dist <= cutoff)
    if np.sum(keep) < 20:
        keep = positive

    angle = np.mod(np.degrees(np.arctan2(dy[keep], dx[keep])), 180.0)
    gamma = 0.5 * (resid[jj[keep]] - resid[ii[keep]]) ** 2

    centers = np.arange(0.0, 180.0, 15.0)
    scores = np.full(len(centers), np.nan)

    for k, c in enumerate(centers):
        diff = np.abs(((angle - c + 90.0) % 180.0) - 90.0)
        sel = diff <= 15.0
        if np.sum(sel) >= 5:
            scores[k] = float(np.median(gamma[sel]))

    if not np.any(np.isfinite(scores)):
        return 0.0, 0.0

    best = int(np.nanargmin(scores))
    phi0 = float(centers[best])

    finite = scores[np.isfinite(scores)]
    contrast = float(np.nanmax(finite) / max(np.nanmin(finite), 1e-12))
    # Conservative start strength; joint PZ-REML determines the final r.
    r_start = float(np.clip(math.sqrt(max(contrast, 1.0)), 1.10, 2.50))
    return z_from_r_phi(r_start, phi0)


def estimate_pz_reml(
    train_locations: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    rng: np.random.Generator,
    max_n: int = PZ_ESTIMATION_MAX_N,
) -> Tuple[float, float, float, bool, float]:
    """
    Frozen PZ-REML pipeline:
        isotropic REML start
        -> geometry-informed start
        -> joint determinant-normalized Z-REML
        -> fixed mild penalty
        -> recover r_hat, phi_hat

    Returns:
        r_hat, phi_hat_deg, kappa_hat, success, objective
    """
    sub = balanced_subset_indices(train_locations, max_n=max_n, rng=rng)
    loc = train_locations[sub]
    y = train_y[sub]
    X = np.column_stack([np.ones(len(sub)), train_x[sub]])

    iso = _isotropic_start(loc, y, X)
    z1g, z2g = _geometry_informed_z_start(loc, y, X)

    kmax = math.log(PZ_R_MAX)
    bounds = [
        (math.log(PZ_RANGE_BOUNDS[0]), math.log(PZ_RANGE_BOUNDS[1])),
        (math.log(PZ_NUGGET_RATIO_BOUNDS[0]), math.log(PZ_NUGGET_RATIO_BOUNDS[1])),
        (-kmax, kmax),
        (-kmax, kmax),
    ]

    starts = [
        iso.copy(),
        np.array([iso[0], iso[1], z1g, z2g], dtype=float),
    ]

    # Additional weak starts improve robustness without tuning to the outcome.
    for phi in (0.0, 45.0, 90.0, 135.0):
        z1, z2 = z_from_r_phi(1.35, phi)
        starts.append(np.array([iso[0], iso[1], z1, z2], dtype=float))

    best = None
    for start in starts:
        opt = minimize(
            _profile_reml_nll,
            start,
            args=(loc, y, X, PZ_LAMBDA_KAPPA, PZ_R_MAX),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": PZ_MAXITER, "ftol": PZ_FTOL},
        )
        val = float(opt.fun) if np.isfinite(opt.fun) else float("inf")
        if best is None or val < best[0]:
            best = (val, opt)

    if best is None:
        return 1.0, 0.0, 0.0, False, float("nan")

    val, opt = best
    z1, z2 = float(opt.x[2]), float(opt.x[3])
    r_hat, phi_hat, kappa_hat = r_phi_from_z(z1, z2)

    # Respect the declared upper bound exactly.
    if r_hat > PZ_R_MAX:
        r_hat = PZ_R_MAX
        kappa_hat = math.log(PZ_R_MAX)

    success = bool(opt.success and np.isfinite(val))
    return r_hat, phi_hat, kappa_hat, success, float(val)


# ---------------------------------------------------------------------
# RF/RFsp/D-RFsp prediction
# ---------------------------------------------------------------------

def make_reference_points(grid_size: int = REFERENCE_GRID_SIZE) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def build_features(
    locations: np.ndarray,
    x_nonspatial: np.ndarray,
    refs: np.ndarray,
    mode: str,
    r: float = 1.0,
    phi_deg: float = 0.0,
) -> np.ndarray:
    if mode == "rf":
        return x_nonspatial
    if mode == "rfsp":
        d = determinant_normalized_distances(locations, refs, 1.0, 0.0)
        return np.column_stack([x_nonspatial, d])
    if mode == "drfsp":
        d = determinant_normalized_distances(locations, refs, r, phi_deg)
        return np.column_stack([x_nonspatial, d])
    raise ValueError(f"Unknown mode: {mode}")


def fit_predict_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        random_state=seed,
        n_jobs=1,  # outer parallelism is handled by joblib
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


# ---------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------

def seed_for(master_seed: int, scenario: Scenario, repetition: int) -> int:
    msg = f"{master_seed}|{scenario.key}|rep={repetition}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def run_replication(
    scenario: Scenario,
    repetition: int,
    master_seed: int,
    include_oracle: bool,
    estimation_max_n: int,
) -> ReplicationResult:
    seed = seed_for(master_seed, scenario, repetition)
    rng = np.random.default_rng(seed)

    locations, x, y = generate_dataset(
        rng,
        n=scenario.n,
        r0=scenario.r0,
        phi0_deg=scenario.phi0_deg,
        noise_sd=scenario.noise_sd,
    )
    train_idx, test_idx = spatial_block_split(locations)

    tr_loc, te_loc = locations[train_idx], locations[test_idx]
    tr_x, te_x = x[train_idx], x[test_idx]
    tr_y, te_y = y[train_idx], y[test_idx]

    # Estimate geometry strictly from training data.
    est_rng = np.random.default_rng((seed + 104729) % (2**32 - 1))
    r_hat, phi_hat, kappa_hat, reml_success, reml_obj = estimate_pz_reml(
        tr_loc, tr_x, tr_y, est_rng, max_n=estimation_max_n
    )

    refs = make_reference_points()

    # RF
    Xtr_rf = build_features(tr_loc, tr_x, refs, "rf")
    Xte_rf = build_features(te_loc, te_x, refs, "rf")
    pred_rf = fit_predict_rf(Xtr_rf, tr_y, Xte_rf, seed + 11)
    rf_rmse, rf_mae, rf_r2 = regression_metrics(te_y, pred_rf)

    # RFsp
    Xtr_rfsp = build_features(tr_loc, tr_x, refs, "rfsp")
    Xte_rfsp = build_features(te_loc, te_x, refs, "rfsp")
    pred_rfsp = fit_predict_rf(Xtr_rfsp, tr_y, Xte_rfsp, seed + 13)
    rfsp_rmse, rfsp_mae, rfsp_r2 = regression_metrics(te_y, pred_rfsp)

    # Estimated D-RFsp
    Xtr_d = build_features(tr_loc, tr_x, refs, "drfsp", r_hat, phi_hat)
    Xte_d = build_features(te_loc, te_x, refs, "drfsp", r_hat, phi_hat)
    pred_d = fit_predict_rf(Xtr_d, tr_y, Xte_d, seed + 17)
    drfsp_rmse, drfsp_mae, drfsp_r2 = regression_metrics(te_y, pred_d)

    # Oracle D-RFsp
    if include_oracle:
        Xtr_o = build_features(
            tr_loc, tr_x, refs, "drfsp", scenario.r0, scenario.phi0_deg
        )
        Xte_o = build_features(
            te_loc, te_x, refs, "drfsp", scenario.r0, scenario.phi0_deg
        )
        pred_o = fit_predict_rf(Xtr_o, tr_y, Xte_o, seed + 19)
        oracle_rmse, oracle_mae, oracle_r2 = regression_metrics(te_y, pred_o)
    else:
        oracle_rmse = oracle_mae = oracle_r2 = float("nan")

    return ReplicationResult(
        scenario_key=scenario.key,
        repetition=repetition,
        seed=seed,
        r_hat=r_hat,
        phi_hat_deg=phi_hat,
        kappa_hat=kappa_hat,
        reml_success=reml_success,
        reml_objective=reml_obj,
        rf_rmse=rf_rmse,
        rf_mae=rf_mae,
        rf_r2=rf_r2,
        rfsp_rmse=rfsp_rmse,
        rfsp_mae=rfsp_mae,
        rfsp_r2=rfsp_r2,
        drfsp_rmse=drfsp_rmse,
        drfsp_mae=drfsp_mae,
        drfsp_r2=drfsp_r2,
        oracle_rmse=oracle_rmse,
        oracle_mae=oracle_mae,
        oracle_r2=oracle_r2,
    )


# ---------------------------------------------------------------------
# Aggregation and markdown output
# ---------------------------------------------------------------------

def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return m, sd


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def scenario_from_key(key: str) -> Scenario:
    parts = {}
    for item in key.split("|"):
        k, v = item.split("=", 1)
        parts[k] = v
    return Scenario(
        n=int(parts["n"]),
        r0=float(parts["r"]),
        phi0_deg=float(parts["phi"]),
        noise_sd=float(parts["noise"]),
    )


def aggregate_recovery(
    scenario: Scenario,
    reps: Sequence[ReplicationResult],
) -> Dict[str, float]:
    rhat = np.asarray([x.r_hat for x in reps], dtype=float)
    mean_r = float(np.mean(rhat))
    sd_r = float(np.std(rhat, ddof=1)) if len(rhat) > 1 else 0.0
    bias = mean_r - scenario.r0
    rmse_r = float(np.sqrt(np.mean((rhat - scenario.r0) ** 2)))

    if abs(scenario.r0 - 1.0) < 1e-12:
        phi_mae = float("nan")
        phi_rmse = float("nan")
    else:
        err = np.asarray(
            [axial_error_deg(x.phi_hat_deg, scenario.phi0_deg) for x in reps],
            dtype=float,
        )
        phi_mae = float(np.mean(err))
        phi_rmse = float(np.sqrt(np.mean(err * err)))

    success_rate = float(np.mean([bool(x.reml_success) for x in reps]))

    return dict(
        mean_r=mean_r,
        sd_r=sd_r,
        bias_r=bias,
        rmse_r=rmse_r,
        phi_mae=phi_mae,
        phi_rmse=phi_rmse,
        success_rate=success_rate,
    )


def aggregate_model(
    reps: Sequence[ReplicationResult],
    prefix: str,
) -> Dict[str, float]:
    rmse = np.asarray([getattr(x, f"{prefix}_rmse") for x in reps], dtype=float)
    mae = np.asarray([getattr(x, f"{prefix}_mae") for x in reps], dtype=float)
    r2 = np.asarray([getattr(x, f"{prefix}_r2") for x in reps], dtype=float)

    return dict(
        mean_rmse=float(np.nanmean(rmse)),
        sd_rmse=float(np.nanstd(rmse, ddof=1)) if np.sum(np.isfinite(rmse)) > 1 else 0.0,
        mean_mae=float(np.nanmean(mae)),
        mean_r2=float(np.nanmean(r2)),
    )


def markdown_config(
    args: argparse.Namespace,
    scenarios: Sequence[Scenario],
) -> str:
    return f"""# D-RFsp Final Manuscript Simulation Results

Generated by `final_simulation.py`.

## Frozen simulation design

- **Two-stage analysis:** Simulation I = anisotropy-parameter recovery; Simulation II = predictive comparison.
- **Replications per scenario:** B = {args.replications}
- **True anisotropy ratios:** {", ".join(map(str, args.r_values))}
- **Axial orientations (degrees):** {", ".join(map(str, args.phi_values))}
- **Sample sizes:** {", ".join(map(str, args.n_values))}
- **Noise/nugget SD levels:** {", ".join(map(str, args.noise_values))}
- **Spatial variance:** {SPATIAL_VARIANCE}
- **Spatial range:** {SPATIAL_RANGE}
- **PZ-REML geometry:** determinant-normalized; Z=(kappa cos(2 phi), kappa sin(2 phi)); kappa=log(r)
- **PZ penalty:** lambda_kappa * ||Z||, lambda_kappa = {PZ_LAMBDA_KAPPA:.8f}
- **PZ r upper bound:** {PZ_R_MAX}
- **Maximum training observations used in covariance estimation:** {args.estimation_max_n}
- **Blocked validation:** top-right spatial quadrant held out
- **RF trees:** {RF_N_ESTIMATORS}
- **Reference grid:** {REFERENCE_GRID_SIZE} x {REFERENCE_GRID_SIZE}
- **Oracle D-RFsp:** {"included" if args.oracle else "not included"}
- **Master seed:** {args.seed}
- **Total scenarios:** {len(scenarios)}
- **Total requested replications:** {len(scenarios) * args.replications}

**Leakage rule:** in every replication, r and phi are estimated from the training data only. Held-out outcomes are never used for anisotropy estimation or tuning.

**Axial-angle note:** phi and phi+180 degrees are equivalent. At r=1 the orientation is not identifiable, so phi-error summaries are reported as `NA`.

"""


def build_markdown(
    args: argparse.Namespace,
    scenarios: Sequence[Scenario],
    results_by_scenario: Dict[str, List[ReplicationResult]],
    started_at: str,
    elapsed_seconds: float,
) -> str:
    out = [markdown_config(args, scenarios)]

    completed = sum(len(v) for v in results_by_scenario.values())
    total = len(scenarios) * args.replications
    out.append("## Progress\n")
    out.append(f"- Started: {started_at}\n")
    out.append(f"- Completed replications: {completed} / {total}\n")
    out.append(f"- Elapsed seconds: {elapsed_seconds:.1f}\n")
    out.append(f"- Completed scenarios: {sum(len(results_by_scenario.get(s.key, [])) >= args.replications for s in scenarios)} / {len(scenarios)}\n\n")

    out.append("## Simulation I — Parameter recovery\n\n")
    out.append(
        "| n | noise SD | true r | true phi (deg) | B complete | mean r_hat | SD(r_hat) | bias(r_hat) | RMSE(r_hat) | axial MAE phi (deg) | axial RMSE phi (deg) | PZ-REML success |\n"
    )
    out.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    for s in scenarios:
        reps = results_by_scenario.get(s.key, [])
        if not reps:
            continue
        a = aggregate_recovery(s, reps)
        out.append(
            f"| {s.n} | {s.noise_sd:g} | {s.r0:g} | {s.phi0_deg:g} | {len(reps)} | "
            f"{fmt(a['mean_r'])} | {fmt(a['sd_r'])} | {fmt(a['bias_r'])} | {fmt(a['rmse_r'])} | "
            f"{fmt(a['phi_mae'], 3)} | {fmt(a['phi_rmse'], 3)} | {fmt(100*a['success_rate'], 1)}% |\n"
        )

    out.append("\n## Simulation II — Predictive comparison\n\n")
    out.append(
        "| n | noise SD | true r | true phi | model | B complete | mean RMSE | SD RMSE | mean MAE | mean R2 | Delta RMSE vs RFsp |\n"
    )
    out.append(
        "|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|\n"
    )

    for s in scenarios:
        reps = results_by_scenario.get(s.key, [])
        if not reps:
            continue

        rfsp = aggregate_model(reps, "rfsp")
        model_specs = [
            ("RF", "rf"),
            ("RFsp", "rfsp"),
            ("D-RFsp estimated", "drfsp"),
        ]
        if args.oracle:
            model_specs.append(("D-RFsp oracle", "oracle"))

        for model_name, prefix in model_specs:
            a = aggregate_model(reps, prefix)
            delta = a["mean_rmse"] - rfsp["mean_rmse"]
            out.append(
                f"| {s.n} | {s.noise_sd:g} | {s.r0:g} | {s.phi0_deg:g} | {model_name} | {len(reps)} | "
                f"{fmt(a['mean_rmse'])} | {fmt(a['sd_rmse'])} | {fmt(a['mean_mae'])} | "
                f"{fmt(a['mean_r2'])} | {fmt(delta)} |\n"
            )

    out.append("\n## Interpretation guide\n\n")
    out.append(
        "- Negative **Delta RMSE vs RFsp** means lower RMSE than RFsp (improvement); positive values mean worse RMSE.\n"
        "- Compare **oracle D-RFsp** with **estimated D-RFsp** to separate available anisotropic predictive benefit from geometry-estimation loss.\n"
        "- Inspect recovery first, then prediction. Poor recovery under high nugget or near isotropy indicates weak identifiability rather than a failure of the nesting property.\n"
        "- At true r=1, phi is not identifiable and should not be interpreted.\n"
    )

    out.append("\n## Minimum manuscript outputs represented by this file\n\n")
    out.append(
        "This markdown contains the numerical material needed for: "
        "(i) estimator bias/RMSE versus true r; "
        "(ii) axial error versus r; "
        "(iii) predictive Delta RMSE versus r; "
        "(iv) parameter-recovery Table 1; and "
        "(v) predictive-comparison Table 2.\n"
    )

    return "".join(out)


def save_checkpoint_line(path: Path, result: ReplicationResult) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), allow_nan=True) + "\n")


def load_checkpoint(path: Path) -> Dict[str, List[ReplicationResult]]:
    out: Dict[str, List[ReplicationResult]] = {}
    if not path.exists():
        return out

    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                key = (d["scenario_key"], int(d["repetition"]))
                if key in seen:
                    continue
                seen.add(key)
                rr = ReplicationResult(**d)
                out.setdefault(rr.scenario_key, []).append(rr)
            except Exception:
                continue

    for key in out:
        out[key].sort(key=lambda z: z.repetition)
    return out


def write_results(
    results_path: Path,
    args: argparse.Namespace,
    scenarios: Sequence[Scenario],
    results_by_scenario: Dict[str, List[ReplicationResult]],
    started_at: str,
    t0: float,
) -> None:
    md = build_markdown(
        args,
        scenarios,
        results_by_scenario,
        started_at,
        time.time() - t0,
    )
    tmp = results_path.with_suffix(results_path.suffix + ".tmp")
    tmp.write_text(md, encoding="utf-8")
    tmp.replace(results_path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final two-stage D-RFsp manuscript simulation."
    )
    p.add_argument("--replications", type=int, default=DEFAULT_B)
    p.add_argument("--r-values", nargs="+", type=float, default=list(DEFAULT_R_VALUES))
    p.add_argument("--phi-values", nargs="+", type=float, default=list(DEFAULT_PHI_VALUES))
    p.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    p.add_argument("--noise-values", nargs="+", type=float, default=list(DEFAULT_NOISE_VALUES))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--jobs", type=int, default=-1, help="joblib jobs; -1 uses all cores")
    p.add_argument(
        "--estimation-max-n",
        type=int,
        default=PZ_ESTIMATION_MAX_N,
        help="Spatially balanced training subset cap for PZ-REML.",
    )
    p.add_argument(
        "--results-file",
        type=str,
        default=RESULTS_FILE,
        help="Markdown output file.",
    )
    p.add_argument(
        "--checkpoint-file",
        type=str,
        default=CHECKPOINT_FILE,
        help="Internal JSONL checkpoint used for resume.",
    )
    p.add_argument(
        "--no-oracle",
        dest="oracle",
        action="store_false",
        help="Disable optional oracle D-RFsp.",
    )
    p.set_defaults(oracle=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint file.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny sanity check instead of the full manuscript design.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.replications < 1:
        raise ValueError("--replications must be >= 1")
    if any(r < 1.0 for r in args.r_values):
        raise ValueError("All r values must be >= 1")
    if any(n < 80 for n in args.n_values):
        raise ValueError("All sample sizes must be >= 80 for the blocked split")
    if any(v <= 0 for v in args.noise_values):
        raise ValueError("Noise SD values must be > 0")
    if args.estimation_max_n < 30:
        raise ValueError("--estimation-max-n must be >= 30")


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        args.replications = 1
        args.r_values = [1.0, 3.0]
        args.phi_values = [30.0]
        args.n_values = [120]
        args.noise_values = [0.5]
        args.estimation_max_n = min(args.estimation_max_n, 70)
        args.jobs = 1

    validate_args(args)

    scenarios = [
        Scenario(n=n, r0=r, phi0_deg=phi, noise_sd=noise)
        for n in args.n_values
        for noise in args.noise_values
        for r in args.r_values
        for phi in args.phi_values
    ]

    results_path = Path(args.results_file).resolve()
    checkpoint_path = Path(args.checkpoint_file).resolve()

    if not args.resume:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        results_by_scenario: Dict[str, List[ReplicationResult]] = {}
    else:
        results_by_scenario = load_checkpoint(checkpoint_path)

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    print("=" * 78)
    print("D-RFsp FINAL MANUSCRIPT SIMULATION")
    print("=" * 78)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Replications/scenario: {args.replications}")
    print(f"Total requested replications: {len(scenarios) * args.replications}")
    print(f"Jobs: {args.jobs}")
    print(f"Results: {results_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    if not args.smoke_test and len(scenarios) * args.replications >= 10000:
        print(
            "NOTE: This is a large manuscript run. Exact GP simulation plus repeated "
            "PZ-REML can take substantial time. The markdown file is updated after "
            "each scenario and the JSONL checkpoint permits --resume."
        )
        print()

    # Write initial file immediately.
    write_results(
        results_path, args, scenarios, results_by_scenario, started_at, t0
    )

    try:
        for s_idx, scenario in enumerate(scenarios, start=1):
            existing = {
                r.repetition
                for r in results_by_scenario.get(scenario.key, [])
            }
            missing = [
                rep for rep in range(args.replications)
                if rep not in existing
            ]

            print(
                f"[{s_idx:03d}/{len(scenarios):03d}] "
                f"{scenario.key} | completed={len(existing)} "
                f"| missing={len(missing)}"
            )

            if not missing:
                continue

            if args.jobs == 1:
                new_results = []
                for rep in missing:
                    rr = run_replication(
                        scenario,
                        rep,
                        args.seed,
                        args.oracle,
                        args.estimation_max_n,
                    )
                    new_results.append(rr)
                    save_checkpoint_line(checkpoint_path, rr)
                    if (rep + 1) % max(1, min(10, args.replications)) == 0:
                        print(
                            f"    rep {rep + 1}/{args.replications} "
                            f"r_hat={rr.r_hat:.3f} "
                            f"phi_hat={rr.phi_hat_deg:.1f} "
                            f"RMSE(est)={rr.drfsp_rmse:.4f}"
                        )
            else:
                # Run this scenario in parallel; checkpoint after collection.
                new_results = Parallel(n_jobs=args.jobs, verbose=0)(
                    delayed(run_replication)(
                        scenario,
                        rep,
                        args.seed,
                        args.oracle,
                        args.estimation_max_n,
                    )
                    for rep in missing
                )
                for rr in new_results:
                    save_checkpoint_line(checkpoint_path, rr)

            results_by_scenario.setdefault(scenario.key, []).extend(new_results)
            results_by_scenario[scenario.key].sort(key=lambda z: z.repetition)

            # Refresh manuscript markdown after every scenario.
            write_results(
                results_path,
                args,
                scenarios,
                results_by_scenario,
                started_at,
                t0,
            )

            a = aggregate_recovery(scenario, results_by_scenario[scenario.key])
            pred = aggregate_model(results_by_scenario[scenario.key], "drfsp")
            print(
                f"    -> mean r_hat={a['mean_r']:.3f}, "
                f"RMSE(r_hat)={a['rmse_r']:.3f}, "
                f"mean D-RFsp RMSE={pred['mean_rmse']:.4f}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Writing current markdown before exit...")
        write_results(
            results_path,
            args,
            scenarios,
            results_by_scenario,
            started_at,
            t0,
        )
        print(f"Progress saved. Resume with: python {Path(__file__).name} --resume")
        return 130

    except Exception:
        print("\nERROR: simulation stopped unexpectedly.")
        traceback.print_exc()
        write_results(
            results_path,
            args,
            scenarios,
            results_by_scenario,
            started_at,
            t0,
        )
        print(f"Partial results saved to: {results_path}")
        print(f"Resume with: python {Path(__file__).name} --resume")
        return 1

    write_results(
        results_path,
        args,
        scenarios,
        results_by_scenario,
        started_at,
        t0,
    )

    print()
    print("=" * 78)
    print("DONE")
    print(f"Final markdown results: {results_path}")
    print(f"Elapsed: {(time.time() - t0) / 60.0:.1f} minutes")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

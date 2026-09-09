#!/usr/bin/env python3
"""Monte Carlo proof-of-concept for RF, RFsp, and D-RFsp.

Source-fixed elements from the supplied D-RFsp document:
- 350 locations sampled uniformly on the unit square.
- Y = X1 + 0.5*X2 + W(s) + epsilon.
- W(s) is a Gaussian spatial effect with anisotropic exponential covariance.
- Principal direction phi = 30 degrees.
- D-RFsp uses the true (phi, r): an oracle-geometry proof of concept.
- Validation is spatially blocked.
- D-RFsp reduces exactly to RFsp at r = 1.

The document does not specify several numerical simulation details. Those are
therefore fixed explicitly below under IMPLEMENTATION-FIXED SETTINGS so the
experiment is reproducible and can be repeated 1, 500, 10,000, ... times.

Examples
--------
    python drfsp_simulation.py --simulations 1
    python drfsp_simulation.py --simulations 500 --jobs -1
    python drfsp_simulation.py --simulations 10000 --jobs -1
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tabulate import tabulate

N_LOCATIONS = 350
PHI_DEGREES = 30.0
R_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0)

SPATIAL_VARIANCE = 1.0       
SPATIAL_RANGE = 0.25         
NOISE_SD = 0.50              
X1_SD = 1.0                  
X2_SD = 1.0                 

REFERENCE_GRID_SIZE = 5

BLOCK_X_CUTOFF = 0.50
BLOCK_Y_CUTOFF = 0.50

RF_N_ESTIMATORS = 20
RF_MAX_DEPTH = 18
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = 0.70

DEFAULT_SEED = 20260909
CHOLESKY_JITTER = 1e-10


@dataclass(frozen=True)
class SimulationResult:
    """RMSE values from one independently generated dataset at one true r."""

    repetition: int
    r: float
    rf_rmse: float
    rfsp_rmse: float
    drfsp_rmse: float

def make_reference_points(grid_size: int = REFERENCE_GRID_SIZE) -> np.ndarray:
    """Create a fixed regular grid of spatial reference locations on [0, 1]^2."""
    axis = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def euclidean_distances(points: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Distances used by ordinary RFsp."""
    delta = points[:, None, :] - refs[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def anisotropic_distances(
    points: np.ndarray,
    refs: np.ndarray,
    phi_degrees: float,
    r: float,
) -> np.ndarray:
    """D-RFsp distance d_A = sqrt(u^2 + r^2 v^2).

    At r=1 we deliberately call the Euclidean implementation so RFsp and
    D-RFsp are numerically identical, matching the exact nesting result.
    """
    if np.isclose(r, 1.0):
        return euclidean_distances(points, refs)

    phi = math.radians(phi_degrees)
    delta = points[:, None, :] - refs[None, :, :]
    dx = delta[:, :, 0]
    dy = delta[:, :, 1]

    u = dx * math.cos(phi) + dy * math.sin(phi)
    v = -dx * math.sin(phi) + dy * math.cos(phi)
    return np.sqrt(u * u + (r * r) * v * v)


def pairwise_anisotropic_distances(
    points: np.ndarray,
    phi_degrees: float,
    r: float,
) -> np.ndarray:
    """Pairwise anisotropic distances among simulated locations."""
    return anisotropic_distances(points, points, phi_degrees, r)


def spatial_covariance(
    points: np.ndarray,
    phi_degrees: float,
    r: float,
    variance: float = SPATIAL_VARIANCE,
    spatial_range: float = SPATIAL_RANGE,
) -> np.ndarray:
    """Anisotropic exponential covariance: sigma^2 exp(-d_A / range)."""
    d = pairwise_anisotropic_distances(points, phi_degrees, r)
    cov = variance * np.exp(-d / spatial_range)
    # Numerical stabilization only; not part of the stochastic model.
    cov.flat[:: cov.shape[0] + 1] += CHOLESKY_JITTER
    return cov

def generate_dataset(rng: np.random.Generator, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one n=350 proof-of-concept dataset for a specified true r."""
    locations = rng.uniform(0.0, 1.0, size=(N_LOCATIONS, 2))
    x1 = rng.normal(0.0, X1_SD, size=N_LOCATIONS)
    x2 = rng.normal(0.0, X2_SD, size=N_LOCATIONS)

    cov = spatial_covariance(locations, PHI_DEGREES, r)
    z = rng.normal(size=N_LOCATIONS)
    spatial_effect = np.linalg.cholesky(cov) @ z
    epsilon = rng.normal(0.0, NOISE_SD, size=N_LOCATIONS)

    y = x1 + 0.5 * x2 + spatial_effect + epsilon
    x_nonspatial = np.column_stack([x1, x2])
    return locations, x_nonspatial, y


def spatial_block_split(locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out the top-right spatial quadrant as the test block."""
    test_mask = (
        (locations[:, 0] >= BLOCK_X_CUTOFF)
        & (locations[:, 1] >= BLOCK_Y_CUTOFF)
    )
    train_mask = ~test_mask

    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(train_idx) < 20 or len(test_idx) < 20:
        raise RuntimeError(
            f"Spatial split produced too few observations: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )
    return train_idx, test_idx

def build_feature_sets(
    locations: np.ndarray,
    x_nonspatial: np.ndarray,
    r: float,
    refs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct RF, RFsp, and D-RFsp feature matrices."""
    rf_features = x_nonspatial

    d_euclidean = euclidean_distances(locations, refs)
    rfsp_features = np.column_stack([x_nonspatial, d_euclidean])

    d_aniso = anisotropic_distances(locations, refs, PHI_DEGREES, r)
    drfsp_features = np.column_stack([x_nonspatial, d_aniso])

    return rf_features, rfsp_features, drfsp_features


def make_rf(seed: int) -> RandomForestRegressor:
    """Create the common Random Forest engine used by all three methods."""
    return RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        random_state=seed,
        n_jobs=1, 
    )


def fit_and_rmse(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> float:
    """Fit one RF and return blocked-test RMSE."""
    model = make_rf(seed)
    model.fit(features[train_idx], y[train_idx])
    pred = model.predict(features[test_idx])
    return float(np.sqrt(mean_squared_error(y[test_idx], pred)))


def run_one_simulation(repetition: int, r: float, master_seed: int) -> SimulationResult:
    """Run one independent simulation for one anisotropy ratio."""
    r_code = int(round(r * 1000))
    seed_sequence = np.random.SeedSequence([master_seed, repetition, r_code])
    data_seed, rf_seed_seq = seed_sequence.spawn(2)
    rng = np.random.default_rng(data_seed)
    rf_seed = int(rf_seed_seq.generate_state(1, dtype=np.uint32)[0])

    locations, x_nonspatial, y = generate_dataset(rng, r)
    train_idx, test_idx = spatial_block_split(locations)
    refs = make_reference_points()
    rf_x, rfsp_x, drfsp_x = build_feature_sets(locations, x_nonspatial, r, refs)

    rf_rmse = fit_and_rmse(rf_x, y, train_idx, test_idx, rf_seed)
    rfsp_rmse = fit_and_rmse(rfsp_x, y, train_idx, test_idx, rf_seed)

    if np.isclose(r, 1.0):
        drfsp_rmse = rfsp_rmse
    else:
        drfsp_rmse = fit_and_rmse(drfsp_x, y, train_idx, test_idx, rf_seed)

    return SimulationResult(
        repetition=repetition,
        r=r,
        rf_rmse=rf_rmse,
        rfsp_rmse=rfsp_rmse,
        drfsp_rmse=drfsp_rmse,
    )


def run_monte_carlo(
    n_simulations: int,
    r_values: Iterable[float],
    master_seed: int,
    jobs: int,
) -> List[SimulationResult]:
    """Run all requested repetitions and anisotropy scenarios."""
    tasks = [
        (rep, r)
        for r in r_values
        for rep in range(1, n_simulations + 1)
    ]

    if jobs == 1:
        results = [run_one_simulation(rep, r, master_seed) for rep, r in tasks]
    else:
        results = Parallel(n_jobs=jobs, verbose=0, batch_size="auto")(
            delayed(run_one_simulation)(rep, r, master_seed) for rep, r in tasks
        )
    return results

def summarize_results(results: List[SimulationResult]) -> List[List[float]]:
    """Average RMSE over repetitions for each r."""
    rows: List[List[float]] = []
    for r in R_VALUES:
        subset = [x for x in results if np.isclose(x.r, r)]
        rf = np.array([x.rf_rmse for x in subset], dtype=float)
        rfsp = np.array([x.rfsp_rmse for x in subset], dtype=float)
        drfsp = np.array([x.drfsp_rmse for x in subset], dtype=float)

        # Positive value means D-RFsp reduced RMSE relative to RFsp.
        gain_pct = 100.0 * (rfsp.mean() - drfsp.mean()) / rfsp.mean()

        rows.append([
            r,
            rf.mean(),
            rfsp.mean(),
            drfsp.mean(),
            gain_pct,
        ])
    return rows


def print_settings(n_simulations: int, jobs: int, seed: int) -> None:
    """Print the complete reproducible experiment configuration."""
    print("\nD-RFsp Monte Carlo proof-of-concept")
    print("=" * 72)
    print(f"Simulations per r       : {n_simulations}")
    print(f"r values                : {list(R_VALUES)}")
    print(f"Locations per dataset   : {N_LOCATIONS}")
    print(f"Principal direction phi : {PHI_DEGREES:.1f} degrees")
    print(f"Spatial variance        : {SPATIAL_VARIANCE}")
    print(f"Spatial range           : {SPATIAL_RANGE}")
    print(f"Noise SD                : {NOISE_SD}")
    print(f"Reference points        : {REFERENCE_GRID_SIZE}x{REFERENCE_GRID_SIZE} grid ({REFERENCE_GRID_SIZE**2})")
    print("Validation block        : x >= 0.50 AND y >= 0.50")
    print(
        "RF settings             : "
        f"trees={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}, "
        f"min_leaf={RF_MIN_SAMPLES_LEAF}, max_features={RF_MAX_FEATURES}"
    )
    print(f"Master seed             : {seed}")
    print(f"Parallel jobs           : {jobs}")
    print("=" * 72)


def print_summary(rows: List[List[float]], elapsed_seconds: float) -> None:
    """Print final averaged results as a clean terminal table."""
    headers = [
        "True r",
        "RF avg RMSE",
        "RFsp avg RMSE",
        "D-RFsp avg RMSE",
        "D-RFsp gain vs RFsp (%)",
    ]
    print("\nAVERAGED BLOCKED-TEST RMSE RESULTS")
    print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=(".2f", ".5f", ".5f", ".5f", ".3f")))
    print(f"\nElapsed time: {elapsed_seconds:.2f} seconds")
    print("Interpretation: positive gain (%) means lower average RMSE for D-RFsp than RFsp.")
    print("Nesting check: at r=1, D-RFsp RMSE is forced to equal RFsp exactly, as proved in the model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated proof-of-concept simulation for RF, RFsp, and D-RFsp."
    )
    parser.add_argument(
        "--simulations",
        "-n",
        type=int,
        default=1,
        help="Number of independent simulations PER r value (e.g. 1, 500, 10000). Default: 1",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Parallel workers across simulations. Use -1 for all CPU cores. Default: 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Master random seed. Default: {DEFAULT_SEED}",
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
        r_values=R_VALUES,
        master_seed=args.seed,
        jobs=args.jobs,
    )
    rows = summarize_results(results)
    elapsed = time.perf_counter() - start
    print_summary(rows, elapsed)


if __name__ == "__main__":
    main()

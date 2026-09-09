# D-RFsp Simulation Experiment Summary

1. The first experiment conducted in this project was the D-RFsp Monte Carlo proof-of-concept simulation. It was run for four simulation counts per anisotropy value: 1, 500, 1000, and 10000. In each run, the same D-RFsp model and the same spatially blocked validation setup were used, while only the number of simulations per r value was changed.

## Standard configuration kept fixed across all simulation runs

- Model: D-RFsp Monte Carlo proof-of-concept
- Number of locations per dataset: 350
- Principal direction: phi = 30 degrees
- Spatial variance: 1.0
- Spatial range: 0.25
- Noise standard deviation: 0.5
- Reference points: 5 x 5 grid (25 points)
- Validation block: x >= 0.50 and y >= 0.50
- Random forest settings: trees = 20, depth = 18, min_leaf = 5, max_features = 0.7
- Master seed: 20260909
- Parallel jobs: 1

## Experimental design used

The experiment evaluated the D-RFsp simulation across the anisotropy values r = 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, and 5.0. The simulation design, data generation process, spatial covariance structure, and model configuration remained standard throughout all runs. Only the simulation count per r was varied between the reported experiments: 1, 500, 1000, and 10000.

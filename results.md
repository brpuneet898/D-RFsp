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

## First experiment results for simulation per r = 1

The first experiment used the D-RFsp Monte Carlo proof-of-concept with one simulation per anisotropy value. The results are shown below for the standard model configuration and fixed experiment settings.

| True r | RF avg RMSE | RFsp avg RMSE | D-RFsp avg RMSE | D-RFsp gain vs RFsp (%) |
| -------- | ------------- | --------------- | ----------------- | -------------------------- |
| 1.00 | 1.01216 | 1.03897 | 1.03897 | 0.000 |
| 1.25 | 0.99659 | 1.14854 | 1.13488 | 1.190 |
| 1.50 | 0.97098 | 0.95251 | 0.90303 | 5.194 |
| 1.75 | 1.38796 | 1.66477 | 1.67240 | -0.459 |
| 2.00 | 1.07443 | 1.07886 | 1.11709 | -3.544 |
| 3.00 | 1.07214 | 1.11250 | 0.99063 | 10.955 |
| 5.00 | 1.04364 | 1.06246 | 1.12483 | -5.870 |

Elapsed time: 1.21 seconds
Interpretation: positive gain (%) means lower average RMSE for D-RFsp than RFsp.

## First experiment results for simulation per r = 500

The first experiment used the D-RFsp Monte Carlo proof-of-concept with 500 simulations per anisotropy value. The results are shown below for the standard model configuration and fixed experiment settings.

| True r | RF avg RMSE | RFsp avg RMSE | D-RFsp avg RMSE | D-RFsp gain vs RFsp (%) |
| -------- | ------------- | --------------- | ----------------- | -------------------------- |
| 1.00 | 1.13578 | 1.10633 | 1.10633 | 0.000 |
| 1.25 | 1.14219 | 1.11098 | 1.10789 | 0.278 |
| 1.50 | 1.15849 | 1.14190 | 1.12937 | 1.097 |
| 1.75 | 1.15800 | 1.14778 | 1.13738 | 0.906 |
| 2.00 | 1.15648 | 1.14513 | 1.12450 | 1.801 |
| 3.00 | 1.17941 | 1.19444 | 1.16093 | 2.805 |
| 5.00 | 1.17916 | 1.20992 | 1.17298 | 3.053 |

Elapsed time: 880.16 seconds

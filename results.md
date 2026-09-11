# D-RFsp Simulation Experiment Summary

## 1

(drfsp_simulation.py)

The first experiment conducted in this project was the D-RFsp Monte Carlo proof-of-concept simulation. It was run for four simulation counts per anisotropy value: 1, 500, 1000, and 10000. In each run, the same D-RFsp model and the same spatially blocked validation setup were used, while only the number of simulations per r value was changed.

### Standard configuration kept fixed across all simulation runs

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

### Experimental design used

The experiment evaluated the D-RFsp simulation across the anisotropy values r = 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, and 5.0. The simulation design, data generation process, spatial covariance structure, and model configuration remained standard throughout all runs. Only the simulation count per r was varied between the reported experiments: 1, 500, 1000, and 10000.

### First experiment results for simulation per r = 1

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

### First experiment results for simulation per r = 500

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

### First experiment results for simulation per r = 1000

The first experiment used the D-RFsp Monte Carlo proof-of-concept with 1000 simulations per anisotropy value. The results are shown below for the standard model configuration and fixed experiment settings.

| True r | RF avg RMSE | RFsp avg RMSE | D-RFsp avg RMSE | D-RFsp gain vs RFsp (%) |
| -------- | ------------- | --------------- | ----------------- | -------------------------- |
| 1.00 | 1.15025 | 1.10448 | 1.10448 | 0.000 |
| 1.25 | 1.15446 | 1.12403 | 1.11905 | 0.443 |
| 1.50 | 1.15694 | 1.14409 | 1.13310 | 0.961 |
| 1.75 | 1.16238 | 1.14979 | 1.14006 | 0.846 |
| 2.00 | 1.16867 | 1.16105 | 1.13812 | 1.976 |
| 3.00 | 1.17527 | 1.19341 | 1.15984 | 2.812 |
| 5.00 | 1.17951 | 1.20833 | 1.17533 | 2.730 |

Elapsed time: 2030.55 seconds

### First experiment results for simulation per r = 10000

The first experiment used the D-RFsp Monte Carlo proof-of-concept with 10000 simulations per anisotropy value. The results are shown below for the standard model configuration and fixed experiment settings.

| True r | RF avg RMSE | RFsp avg RMSE | D-RFsp avg RMSE | D-RFsp gain vs RFsp (%) |
| -------- | ------------- | --------------- | ----------------- | -------------------------- |
| 1.00 | 1.14301 | 1.10471 | 1.10471 | 0.000 |
| 1.25 | 1.15416 | 1.12476 | 1.12177 | 0.266 |
| 1.50 | 1.15809 | 1.13835 | 1.12955 | 0.774 |
| 1.75 | 1.16558 | 1.15155 | 1.13880 | 1.108 |
| 2.00 | 1.16775 | 1.16291 | 1.14472 | 1.564 |
| 3.00 | 1.17603 | 1.18747 | 1.15634 | 2.622 |
| 5.00 | 1.18221 | 1.20771 | 1.17113 | 3.029 |

Elapsed time: 8105.75 seconds

## 2

(conservative_estimated_drfsp_simulation.py)

The second experiment conducted in this project was the Conservative Estimated D-RFsp-I Monte Carlo simulation. This version used an estimated structural anisotropy search together with a conservative safeguard. The experiment was run for two simulation counts per anisotropy value: 1 and 500. In both runs, the same data generation settings, blocked validation setup, structural search grids, predictive candidate grids, random forest settings, and conservative safeguard were used. Only the number of simulations per r value was changed.

### Conservative estimated configuration kept fixed across all simulation runs

- Model: Conservative Estimated D-RFsp-I Monte Carlo simulation
- True r values: 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, and 5.0
- True phi: 30.0 degrees
- Number of locations per dataset: 350
- Spatial variance: 1.0
- Spatial range: 0.25
- Noise standard deviation: 0.5
- Reference points: 5 x 5 grid (25 points)
- Outer validation block: x >= 0.50 and y >= 0.50
- Inner spatial folds: 4
- Structural phi grid: 0, 5, ..., 175 degrees
- Structural r grid: 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, and 6.0
- Predictive r candidates: 1.0, 1.25, 1.5, 2.0, 3.0, and 4.0
- Predictive phi offsets: -15.0, -5.0, 0.0, 5.0, and 15.0 degrees
- Conservative safeguard: use isotropic if L_iso <= L_best + SE_best
- Random forest settings: trees = 20, depth = 18, min_leaf = 5, max_features = 0.7
- Master seed: 20260909
- Parallel jobs: 1

### Experimental design used

The experiment evaluated the conservative estimated D-RFsp-I simulation across the anisotropy values r = 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, and 5.0. The structural anisotropy was estimated using the fixed structural phi and r grids, and predictive evaluation used the fixed predictive r candidates and phi offsets. The conservative safeguard selected the isotropic version when the isotropic loss was within one standard error of the best structural loss. Only the simulation count per r was varied between the reported experiments: 1 and 500.

### Second experiment results for simulation per r = 1

The second experiment used the Conservative Estimated D-RFsp-I Monte Carlo simulation with one simulation per anisotropy value. The results are shown below for the fixed conservative estimated configuration.

| True r | RF avg RMSE | RFsp avg RMSE | D-RFsp avg RMSE | D-RFsp gain vs RFsp (%) |
| -------- | ------------- | --------------- | ----------------- | -------------------------- |
| 1.00 | 1.01728 | 1.00085 | 1.05091 | -5.001 |
| 1.25 | 0.99660 | 1.15850 | 1.15850 | 0.000 |
| 1.50 | 0.98288 | 0.99563 | 0.99563 | 0.000 |
| 1.75 | 1.37118 | 1.65665 | 1.94535 | -17.426 |
| 2.00 | 1.06562 | 1.07251 | 1.07251 | 0.000 |
| 3.00 | 1.08846 | 1.17633 | 1.17633 | 0.000 |
| 5.00 | 1.07149 | 1.02471 | 1.07030 | -4.449 |

Elapsed time: 146.98 seconds
Interpretation: positive gain (%) means lower average RMSE for D-RFsp than RFsp.

### Second experiment results for simulation per r = 500

The second experiment used the Conservative Estimated D-RFsp-I Monte Carlo simulation with 500 simulations per anisotropy value. The results are shown below for the fixed conservative estimated configuration.

| True r | RF avg RMSE | RFsp avg RMSE | D-RFsp avg RMSE | D-RFsp gain vs RFsp (%) |
| -------- | ------------- | --------------- | ----------------- | -------------------------- |
| 1.00 | 1.13492 | 1.10362 | 1.12806 | -2.215 |
| 1.25 | 1.14208 | 1.11029 | 1.13137 | -1.898 |
| 1.50 | 1.15803 | 1.14156 | 1.15236 | -0.946 |
| 1.75 | 1.15891 | 1.14912 | 1.15690 | -0.677 |
| 2.00 | 1.15555 | 1.14382 | 1.16003 | -1.417 |
| 3.00 | 1.17830 | 1.19502 | 1.19433 | 0.058 |
| 5.00 | 1.18067 | 1.20634 | 1.20491 | 0.118 |

Elapsed time: 36175.39 seconds

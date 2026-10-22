# Beijing Multi-Site Air Quality — D-RFsp Real-Data Results

> **Status:** COMPLETE

## Dataset and frozen design

- Source files: 12 PRSA station CSVs; raw hourly records: 420,768.
- Station-day observations with at least 18 observed hourly PM2.5 measurements: 17,164.
- Unique monitoring sites: 12; dates: 2013-03-01 to 2017-02-28.
- Response: daily average PM2.5 (µg/m³). This is a DAILY aggregation of the supplied hourly data.
- Common meteorological inputs: TEMP, PRES, DEWP, RAIN, WSPM; time harmonics (day-of-year and month).
- Other pollutant concentrations (PM10, SO2, NO2, CO, O3) deliberately excluded to avoid pollutant-proxy effects.
- Fold assignment: station-wise four-fold holdout (3 entire stations per fold); fixed alphabetic grouping.
- Same RF engine for every arm: 200 trees, min_samples_leaf=5, max_features=0.70, fixed seed.
- RF = meteorology/time only; RF-XYT = RF + projected station x,y; RFsp = RF + 25 isotropic reference distances; PZ-D-RFsp = identical RFsp features with PZ-estimated directional distances.
- Geometry: determinant-normalized d² = u²/r + rv²; r≥1, axial φ∈[0°,180°).
- PZ: isotropic REML start, geometry start, joint Z-REML, penalty λ||Z|| with λ=-log(.30)/log(3), r_max=10; spatial range [0.02,2], nugget ratio [1e-4,10].
- Coordinates: published approximate station coordinates, locally projected into kilometers, scaled by a common spatial length.
- Screening anisotropy on FULL data is descriptive only; each prediction fold estimates its geometry with its TRAINING stations and outcomes only.
- Missing meteorological features are filled with training-only medians. Scoring uses observed PM2.5 targets only.

## Full-data anisotropy screening (NOT used for held-out prediction)

| n stations | r_hat | phi_hat (degrees) | optimizer success | at r_max |
| --- | --- | --- | --- | --- |
| 12 | 3.6467 | 45.0389 | True | False |


## Model comparison (pooled station-out-of-sample observations)

| model | RMSE | MAE | R² | Delta RMSE vs RFsp (negative=better) |
| --- | --- | --- | --- | --- |
| RF | 25.0706 | 15.9324 | 0.8674 | -0.5431 |
| RF-XYT | 25.3227 | 16.2650 | 0.8647 | -0.2910 |
| RFsp | 25.6137 | 16.5484 | 0.8615 | 0.0000 |
| PZ-D-RFsp | 25.4806 | 16.4119 | 0.8630 | -0.1331 |


## Fold-specific anisotropy and paired predictive differences

| fold | held-out stations | r_hat | phi_hat (°) | REML success | test n | RFsp RMSE | PZ-D-RFsp RMSE | paired Delta RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Aotizhongxin, Changping, Dingling | 3.4280 | 42.3756 | True | 4285 | 26.3139 | 26.4639 | 0.1500 |
| 2 | Dongsi, Guanyuan, Gucheng | 3.9689 | 44.2880 | True | 4299 | 23.6059 | 23.2903 | -0.3156 |
| 3 | Huairou, Nongzhanguan, Shunyi | 2.9067 | 52.2287 | True | 4278 | 28.6038 | 28.4683 | -0.1355 |
| 4 | Tiantan, Wanliu, Wanshouxigong | 4.6142 | 44.9943 | True | 4302 | 23.6090 | 23.3419 | -0.2671 |


## All fold-wise model metrics

| fold | model | test n | RMSE | MAE | R² |
| --- | --- | --- | --- | --- | --- |
| 1 | RF | 4285 | 26.1171 | 17.1572 | 0.8396 |
| 1 | RF-XYT | 4285 | 26.2396 | 17.2693 | 0.8381 |
| 1 | RFsp | 4285 | 26.3139 | 17.3345 | 0.8372 |
| 1 | PZ-D-RFsp | 4285 | 26.4639 | 17.3509 | 0.8353 |
| 2 | RF | 4299 | 22.7716 | 14.1045 | 0.8979 |
| 2 | RF-XYT | 4299 | 23.2198 | 14.7065 | 0.8938 |
| 2 | RFsp | 4299 | 23.6059 | 14.9892 | 0.8902 |
| 2 | PZ-D-RFsp | 4299 | 23.2903 | 14.7216 | 0.8932 |
| 3 | RF | 4278 | 28.2162 | 18.6454 | 0.8246 |
| 3 | RF-XYT | 4278 | 28.4725 | 18.9577 | 0.8214 |
| 3 | RFsp | 4278 | 28.6038 | 19.1257 | 0.8198 |
| 3 | PZ-D-RFsp | 4278 | 28.4683 | 19.0065 | 0.8215 |
| 4 | RF | 4302 | 22.7687 | 13.8410 | 0.8963 |
| 4 | RF-XYT | 4302 | 22.9707 | 14.1446 | 0.8945 |
| 4 | RFsp | 4302 | 23.6090 | 14.7607 | 0.8885 |
| 4 | PZ-D-RFsp | 4302 | 23.3419 | 14.5857 | 0.8911 |


Completed folds: 4/4. Fold wins (negative Delta): 3/4. Mean fold paired Delta RMSE: -0.1421.

## Statistical and identifiability cautions

- ONLY TWELVE distinct spatial coordinates exist; no number of hourly rows increases the number of spatial locations.
- PZ-REML uses adjusted STATION MEAN residuals (one number/site), after removing available covariates, to prevent pseudoreplication at duplicate site coordinates. This preprocessing adaptation differs from the simulation’s independent-point GP setting.
- Fold PZ fitting has only 9 sites; r and φ are weakly identified and confidence claims should be cautious.
- Residuals pooled over time are not a fully specified spatiotemporal likelihood. Global spatial anisotropy estimates are exploratory screening estimates, not precise physical measurements.
- Fold RMSE differences are paired because RFsp and PZ-D-RFsp use exactly the same holdout observations; no single-fold superiority proves a general effect.
- Reported full-data r/φ are never inserted into test-fold D-RFsp distance construction.
- No anisotropy-driven tuning of hyperparameters or selection of folds occurs.

## Provenance and reproducibility

- Dataset: UCI Beijing Multi-Site Air Quality (PRSA), https://archive.ics.uci.edu/dataset/501/beijing
- Station coordinate reference: https://www.mdpi.com/2071-1050/14/9/5104 (Table 1).
- Frozen simulation source for optimizer constants and features: https://github.com/brpuneet898/D-RFsp/blob/main/final_simulation.py
- To reproduce: `python beijing-script.py` from the project folder; save CSVs in `dataset/`.
- No synthetic model scores or manually supplied anisotropy parameters are used.

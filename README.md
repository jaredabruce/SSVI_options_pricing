# Extended SSVI Volatility Surface Project
Author: Jared Bruce
Data: End-of-Day SPY Historical Options from 2020-2022

## 1. Overview
This project demonstrates how to:

1. Clean and preprocess raw options data (e.g., SPY EOD from 2020–2022).
2. Calibrate an Extended SSVI (Stochastic–Spot Volatility Inspired) model to ensure a no-arbitrage implied volatility surface across strikes and maturities.
3. Visualize the resulting surface in 2D and 3D charts.
4. Price options (calls/puts) for given strikes/expiries with the fitted Extended SSVI surface.

## 2. Repository Structure

```
.
├── data_cleaning.py
├── train_extended_ssvi.py
├── visualize_extended_ssvi.ipynb   (or visualize_extended_ssvi.py)
├── price_option_with_extended_ssvi.py
├── requirements.txt                (or Pipfile / pyproject.toml)
├── raw_spy_options_2020_2022.csv   (example raw data file, not included here)
├── options_cleaned.csv             (output after cleaning)
├── extended_ssvi_params.csv        (output after calibration)
└── README.md                       (this file)
```

1. ```data_cleaning.py```
    * Reads a raw CSV of historical options data.
    * Uses tqdm for a progress bar if the file is large.
    * Produces a “long format” CSV named ```options_cleaned.csv```, with columns like:
        * ```quote_date```, ```expire_date```, ```option_type```, ```strike```, ```underlying_last```, ```iv```, ```mid_price```, ```time_to_expiry_years```, etc.

2. ```train_extended_ssvi.py```
    * Reads ```options_cleaned.csv```.
    * Calibrates a global Extended SSVI model (with parameters [a<sub>0</sub>,a<sub>1</sub>, p, n]) across all maturities in the dataset.
    * Ensures no strike arbitrage and partial no calendar arbitrage by enforcing a monotonic 𝜃(𝑇) = ```a0 + a1*T``` (with ```a1 >= 0```).
    * Uses tqdm to show progress building arrays or doing multiple random initial guesses.
    * Outputs ```extended_ssvi_params.csv``` with the fitted parameters.

3. ```visualize_extended_ssvi.py```
    * Loads ```extended_ssvi_params.csv```.
    * Shows 2D and 3D plots of implied volatility vs. strike vs. maturity.

4. ```price_option_with_extended_ssvi.py```
    * Loads ```extended_ssvi_params.csv```.
    * Takes user input (strike, expiry date, etc.).
    * Optionally fetches live spot from yfinance (or you can provide a custom spot).
    * Computes the implied volatility from the Extended SSVI formula and prices the option with Black–Scholes.



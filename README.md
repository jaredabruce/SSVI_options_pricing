# Extended SSVI Volatility Surface
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

## 3 Installation & Setup

1. Clone this repository (or copy the files locally).
2. Install dependencies: ```pip install -r requirements.txt``` or ```pip install pandas numpy scipy yfinance tqdm matplotlib```
3. Obtain or place your raw options CSV in the repository, e.g. ```raw_spy_options_2020_2022.csv```.

## 4 Usage

### 4.1 Data Cleaning

```python data_cleaning.py raw_spy_options_2020_2022.csv```
* Shows a progress bar while reading large files.
* Produces ```options_cleaned.csv```.

### 4.2 Train Extended SSVI

```python train_extended_ssvi.py options_cleaned.csv extended_ssvi_params.csv```
* Reads ```options_cleaned.csv```.
* Calibrates global Extended SSVI parameters.
* Saves them in ```extended_ssvi_params.csv```.

### 4.3 Visualize the Surface

Run ```visualize_extended_ssvi.py``` to produce plots.


![Example 2D Chart (Slices)](/Images/2D.png)

![Example 3D Chart (Surface)](/Images/3d.png)

### 4.4 Price an Option

``` python price_option_with_extended_ssvi.py 420 2024-06-21 C```
* Fetches the latest SPY spot from yfinance (or fallback).
* Uses ```extended_ssvi_params.csv``` to compute the implied volatility at the given strike & maturity.
* Prints out a Black–Scholes call price.
(Adjust the command to specify a custom spot: ```python price_option_with_extended_ssvi.py 420 2024-06-21 C 410``` if you want S = 410.)

## 5. Project Notes & Limitations

1. Data Source: This example uses EOD SPY historical options from 2020–2022.
2. No Dividends: We assume a zero-dividend / basic carry approach. Real equity index modeling typically uses forward prices or dividend assumptions.
3. Simplified Extended SSVI: We parametrize θ(T)=a<sub>0</sub> + a<sub>1</sub>T. Real desks often use piecewise functions or splines for θ(T), ρ(T), η(T).
4. Arbitrage: The single-slice SSVI formula ensures no strike arbitrage, while enforcing a<sub>1</sub> ≥ 0 partially addresses calendar arbitrage. For fully robust calibrations, more advanced monotonic constraints or global fits might be needed.
5. Performance: The scripts use tqdm for progress bars and chunked reading to handle large CSVs. If you have tens of millions of rows, you might consider a database approach or more optimized libraries.

# train_extended_ssvi.py

import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm

###############################################################################
# Extended SSVI param:
#   w(k, T) = 0.5 * [theta(T)] * [1 + rho * (eta*k) + sqrt((eta*k+rho)^2 + 4(1-rho^2)) ]
#   where theta(T) = a0 + a1*T,  a1 >= 0, a0>0,  rho in(-1,1), eta>0
###############################################################################

def extended_ssvi_slice(k, T, a0, a1, rho, eta):
    theta_T = a0 + a1*T
    part = (eta*k + rho)
    return 0.5 * theta_T * (
        1 + rho*(eta*k) + np.sqrt(part**2 + 4*(1-rho**2))
    )

def objective(params, ks, Ts, w_market):
    a0, a1, rho, eta = params
    w_model = extended_ssvi_slice(ks, Ts, a0, a1, rho, eta)
    return w_model - w_market

def train_extended_ssvi(cleaned_csv="options_cleaned.csv", out_csv="extended_ssvi_params.csv"):
    # 1) Read cleaned data
    print(f"Loading cleaned data from {cleaned_csv}...")
    df = pd.read_csv(cleaned_csv, low_memory=False)

    # 2) Filter to calls
    df_calls = df[df["option_type"] == "C"].copy()
    df_calls.dropna(subset=["iv", "strike", "underlying_last", "time_to_expiry_years"], inplace=True)
    df_calls = df_calls[df_calls["iv"] > 0]

    # 3) Build arrays
    #    If you have a huge dataset, you could do row-by-row with tqdm:
    #    But if it's not massive, a vectorized approach is usually instant.
    #    We'll show a small example:
    T_list = []
    k_list = []
    w_list = []

    print("Converting data to arrays...")
    for idx, row in tqdm(df_calls.iterrows(), total=len(df_calls), desc="Building arrays"):
        T = row["time_to_expiry_years"]
        S = row["underlying_last"]
        K = row["strike"]
        iv = row["iv"]
        if iv <= 0 or S<=0 or K<=0 or T<=0:
            continue
        w = iv**2 * T
        k = np.log(K/S)
        w_list.append(w)
        k_list.append(k)
        T_list.append(T)

    ks = np.array(k_list)
    Ts = np.array(T_list)
    w_market = np.array(w_list)
    print(f"Final dataset size: {len(w_market)} implied vol points")

    # 4) We'll do a multi-start approach, so we can show a progress bar
    #    for each random initial guess. This is optional but can help if we suspect local minima.
    N_ATTEMPTS = 3
    best_res = None
    best_cost = 1e15

    print("Fitting extended SSVI with multiple initial guesses...")
    attempts = []
    for i in tqdm(range(N_ATTEMPTS), desc="Global SSVI fit attempts"):
        # random-ish initial guesses
        a0_init = np.random.uniform(0.001, 1.0) * np.mean(w_market)
        a1_init = np.random.uniform(0, 1.0)
        rho_init= np.random.uniform(-0.95, 0.95)
        eta_init= np.random.uniform(0.1, 5.0)

        init_params = np.array([a0_init, a1_init, rho_init, eta_init])
        lb = [1e-6, 0.0, -0.999, 1e-6]
        ub = [10.0, 10.0, 0.999, 100.0]

        def residuals(p):
            return objective(p, ks, Ts, w_market)

        res = least_squares(residuals, init_params, bounds=(lb,ub))
        cost = res.cost
        if cost < best_cost:
            best_cost = cost
            best_res = res
        
        attempts.append((res.x, cost))
    
    # best_res should hold the best solution
    a0_fit, a1_fit, rho_fit, eta_fit = best_res.x
    results = {
        "a0": a0_fit,
        "a1": a1_fit,
        "rho": rho_fit,
        "eta": eta_fit,
        "residual": best_res.cost,
        "num_points": len(w_market)
    }

    # Save to CSV
    df_out = pd.DataFrame([results])
    df_out.to_csv(out_csv, index=False)
    print(f"\nExtended SSVI calibration done. Best cost={best_res.cost:.4f}")
    print(f"Saved params to {out_csv}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_extended_ssvi.py <options_cleaned.csv> [<out_params.csv>]")
        sys.exit(1)
    in_csv = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "extended_ssvi_params.csv"
    
    train_extended_ssvi(in_csv, out_csv)

if __name__ == "__main__":
    main()

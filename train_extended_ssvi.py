# train_extended_ssvi.py

import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

###############################################################################
# 1) Extended SSVI Param: w(k, T) = 0.5 * theta(T) * [1 + rho(T)* (eta(T)*k)
#        + sqrt( (eta(T)*k + rho(T))^2 + 4(1 - rho(T)^2 ) ]
#
# We'll define a simplified model:
#   theta(T) = a0 + a1 * T   (with a1 >= 0, a0 > 0)
#   rho(T)   = const_rho  (in (-1,1))
#   eta(T)   = const_eta  (in > 0)
###############################################################################
def extended_ssvi_slice(k, T, a0, a1, rho, eta):
    """
    Extended SSVI formula for total implied variance (TIV) = w(k,T).
    """
    theta_T = a0 + a1 * T  # must be positive
    part = (eta * k + rho)
    return 0.5 * theta_T * (
        1 + rho * (eta * k) + np.sqrt(part**2 + 4*(1 - rho**2))
    )

def objective(params, ks, Ts, w_market):
    """
    Residual between market TIV and extended SSVI across all data points.
    params = [a0, a1, rho, eta]
    """
    a0, a1, rho, eta = params
    # apply bounds checks manually or rely on 'least_squares' with bounds
    w_model = extended_ssvi_slice(ks, Ts, a0, a1, rho, eta)
    return w_model - w_market

def train_extended_ssvi(cleaned_csv="options_cleaned.csv", out_csv="extended_ssvi_params.csv"):
    """
    1) Load the cleaned CSV.
    2) We'll combine *all* maturities, *all* strikes for calls (option_type=="C"), 
       convert to total implied variance and k=ln(K/S).
    3) We'll do a global fit for [a0, a1, rho, eta].
    4) Save those parameters to a CSV.
    """
    df = pd.read_csv(cleaned_csv)
    
    # Filter to calls for demonstration
    df_calls = df[df["option_type"] == "C"].copy()
    df_calls.dropna(subset=["iv", "strike", "underlying_last", "time_to_expiry_years"], inplace=True)
    df_calls = df_calls[df_calls["iv"] > 0]
    
    # Build arrays of T, k, w_market
    T_array = df_calls["time_to_expiry_years"].values
    S_array = df_calls["underlying_last"].values
    K_array = df_calls["strike"].values
    iv_array = df_calls["iv"].values
    
    # total implied variance
    w_market = iv_array**2 * T_array
    
    # log-moneyness
    ks = np.log(K_array / S_array)
    
    # initial guess
    # a0, a1, rho, eta
    a0_init = np.mean(w_market)  # naive guess
    a1_init = 0.0                # start with 0 slope
    rho_init = 0.0               # no skew tilt
    eta_init = 1.0               # some guess
    
    init_params = np.array([max(a0_init, 1e-3), a1_init, rho_init, eta_init])
    
    # bounds:
    #  a0 > 0
    #  a1 >= 0
    #  rho in (-1, 1)
    #  eta > 0
    lb = [1e-6, 0.0, -0.999, 1e-6]
    ub = [10.0, 10.0, 0.999, 100.0]
    
    def residuals(p):
        return objective(p, ks, T_array, w_market)
    
    res = least_squares(residuals, init_params, bounds=(lb, ub))
    a0_fit, a1_fit, rho_fit, eta_fit = res.x
    
    # We assume these 4 parameters define our entire surface
    # Save them
    results = {
        "a0": a0_fit,
        "a1": a1_fit,
        "rho": rho_fit,
        "eta": eta_fit,
        "residual": res.cost,
        "num_points": len(w_market)
    }
    df_out = pd.DataFrame([results])
    df_out.to_csv(out_csv, index=False)
    print(f"Extended SSVI calibration done. Saved params to {out_csv}")
    print(results)

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_extended_ssvi.py <options_cleaned.csv> [<out_params.csv>]")
        sys.exit(1)
    in_csv = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "extended_ssvi_params.csv"
    
    train_extended_ssvi(in_csv, out_csv)

if __name__ == "__main__":
    main()

# price_option_with_extended_ssvi.py

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime

def black_scholes_price(S, K, T, r, sigma, option_type="C"):
    if T <= 0:
        # immediate expiry => intrinsic
        if option_type.upper() == "C":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if option_type.upper() == "C":
        return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    else:
        return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def extended_ssvi_slice(k, T, a0, a1, rho, eta):
    theta_T = a0 + a1*T
    part = (eta*k + rho)
    return 0.5 * theta_T * (
        1 + rho*(eta*k) + np.sqrt(part**2 + 4*(1-rho**2))
    )

def price_option(
    strike: float,
    expiry_date: str,
    option_type: str="C",
    spot: float=None,
    r: float=0.01,
    extended_ssvi_csv="extended_ssvi_params.csv"
):
    """
    1) Load the Extended SSVI global params from 'extended_ssvi_csv'.
    2) Compute T_live = (expiry_date - today)/365.
    3) If spot=None, fetch from yfinance (SPY).
    4) Evaluate SSVI => implied vol => Black–Scholes => return price.
    """
    # 1) load SSVI params
    df_params = pd.read_csv(extended_ssvi_csv)
    a0 = df_params["a0"].iloc[0]
    a1 = df_params["a1"].iloc[0]
    rho = df_params["rho"].iloc[0]
    eta = df_params["eta"].iloc[0]

    # 2) compute T_live
    today = datetime.now()
    expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    days_to_expiry = (expiry_dt - today).days
    T_live = max(days_to_expiry, 0)/365.0

    # fallback if T_live < some small
    if T_live <= 0:
        # Option is effectively expired, payoff is intrinsic
        if spot is None:
            # fetch spot anyway
            ticker = yf.Ticker("SPY")
            df_spot = ticker.history(period="1d", interval="1m")
            if len(df_spot)>0:
                spot = df_spot["Close"].iloc[-1]
            else:
                spot= strike # fallback
        return black_scholes_price(spot, strike, 0, r, 0, option_type=option_type)

    # 3) if spot not given, fetch from yfinance
    if spot is None:
        ticker = yf.Ticker("SPY")
        df_spot = ticker.history(period="1d", interval="1m")
        if len(df_spot) > 0:
            spot = df_spot["Close"].iloc[-1]
        else:
            # fallback guess
            spot = strike

    # 4) Evaluate SSVI
    k = np.log(strike / spot)
    w_val = extended_ssvi_slice(k, T_live, a0, a1, rho, eta)
    if w_val <= 0:
        # degenerate
        return 0.0
    sigma = sqrt(w_val / T_live)

    # 5) price with BS
    bs_price = black_scholes_price(spot, strike, T_live, r, sigma, option_type=option_type)
    return bs_price

def main():
    if len(sys.argv) < 3:
        print("Usage: python price_option_with_extended_ssvi.py <strike> <expiry_date YYYY-MM-DD> [option_type] [spot]")
        sys.exit(1)

    strike = float(sys.argv[1])
    expiry_date = sys.argv[2]
    option_type = sys.argv[3] if len(sys.argv) > 3 else "C"
    spot_arg = float(sys.argv[4]) if len(sys.argv) > 4 else None

    px = price_option(strike, expiry_date, option_type=option_type, spot=spot_arg)
    print(f"Price of {option_type} with strike={strike} expiry={expiry_date} -> {px:.4f}")

if __name__ == "__main__":
    main()

# visualize_extended_ssvi.ipynb (excerpt)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

def extended_ssvi_slice(k, T, a0, a1, rho, eta):
    theta_T = a0 + a1*T
    part = (eta*k + rho)
    return 0.5 * theta_T * (
        1 + rho*(eta*k) + np.sqrt(part**2 + 4*(1-rho**2))
    )

def main():
    # Load the fitted extended SSVI params
    df_params = pd.read_csv("extended_ssvi_params.csv")
    a0, a1, rho, eta = df_params.iloc[0][["a0","a1","rho","eta"]].values

    # Plot a few slices in T (e.g., 0.1y, 0.5y, 1y, 2y)
    T_slices = [0.1, 0.5, 1.0, 2.0]
    k_grid = np.linspace(-0.6, 0.6, 100)  # log-moneyness range

    plt.figure(figsize=(8,5))
    for T in T_slices:
        w_vals = [extended_ssvi_slice(k, T, a0, a1, rho, eta) for k in k_grid]
        # implied vol = sqrt(w / T)
        iv_vals = [np.sqrt(w/T) for w in w_vals]
        plt.plot(k_grid, iv_vals, label=f"T={T}y")

    plt.title("Extended SSVI: Implied Vol slices")
    plt.xlabel("Log-moneyness k")
    plt.ylabel("Implied Vol")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    k_grid = np.linspace(-1.0, 1.0, 50)
    T_grid = np.linspace(0.05, 2.0, 50)

    K, TT = np.meshgrid(k_grid, T_grid)
    W = extended_ssvi_slice(K, TT, a0, a1, rho, eta)
    IV = np.sqrt(W / TT)

    ax.plot_surface(K, TT, IV, cmap='viridis')
    ax.set_xlabel("k (log-moneyness)")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("Implied Vol")
    ax.set_title("Extended SSVI Implied Vol Surface")
    plt.show()

if __name__ == "__main__":
    main()

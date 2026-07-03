import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_surfaces(
        model,
        datas,
        dataf,
        my_predicts,
        dafolder,
        figformat,
        deflections,
        ngrid=100):

    """
    Creates one 3D surface (Alpha x Beta -> CH) for each deflection.
    Training points are shown as black scatter.
    """

    nplots = len(deflections)
    ncols = 3
    nrows = int(np.ceil(nplots / ncols))

    fig = plt.figure(figsize=(6*ncols, 5*nrows))

    alpha = np.linspace(datas["alpha"].min(),
                        datas["alpha"].max(),
                        ngrid)

    beta = np.linspace(datas["beta"].min(),
                       datas["beta"].max(),
                       ngrid)

    AA, BB = np.meshgrid(alpha, beta)

    for i, d in enumerate(deflections):

        ax = fig.add_subplot(nrows, ncols, i+1,
                             projection="3d")

        X = pd.DataFrame({
            "alpha": AA.ravel(),
            "beta": BB.ravel(),
            "deflection": d
        })

        CH = my_predicts(model, X.to_numpy()).reshape(AA.shape)

        ax.plot_surface(
            AA,
            BB,
            CH,
            cmap=cm.viridis,
            linewidth=0,
            alpha=0.7)

        mask = np.isclose(datas["deflection"], d)

        ax.scatter(
            datas.loc[mask, "alpha"],
            datas.loc[mask, "beta"],
            dataf.loc[mask],
            color="k",
            s=10)

        ax.set_title(f"Deflection = {d:.1f}°")

        ax.set_xlabel("Alpha")
        ax.set_ylabel("Beta")
        ax.set_zlabel("CH")

        ax.view_init(elev=20, azim=135)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            dafolder,
            "CH_surfaces." + figformat),
        dpi=400)

    plt.close()


def plot_ch_vs_deflection(
        model,
        datas,
        dataf,
        my_predicts,
        dafolder,
        figformat,
        alphas,
        beta_fixed=0.0,
        alpha_tol=0.25,
        beta_tol=0.25):

    """
    CH vs deflection.
    One curve for each alpha.
    Beta is kept fixed.
    """

    plt.figure(figsize=(10,7))

    x = np.linspace(
        datas["deflection"].min(),
        datas["deflection"].max(),
        300)

    for alpha0 in alphas:

        mask = (
            np.isclose(datas["alpha"], alpha0, atol=alpha_tol)
            &
            np.isclose(datas["beta"], beta_fixed, atol=beta_tol)
        )

        if mask.sum() == 0:
            continue

        X = pd.DataFrame({
            "alpha": np.full_like(x, alpha0),
            "beta": np.full_like(x, beta_fixed),
            "deflection": x
        })

        y = my_predicts(model, X.to_numpy())

        plt.plot(
            x,
            y,
            lw=1.5,
            label=f"α={alpha0:.0f}°")

        plt.scatter(
            datas.loc[mask,"deflection"],
            dataf.loc[mask],
            s=20)

    plt.grid(True)
    plt.xlabel("Deflection (deg)")
    plt.ylabel("CH")
    plt.legend()

    plt.savefig(
        os.path.join(
            dafolder,
            "CH_vs_Deflection."+figformat),
        dpi=400)

    plt.close()


def plot_ch_vs_beta(
        model,
        datas,
        dataf,
        my_predicts,
        dafolder,
        figformat,
        deflections,
        alpha_fixed=0.0,
        alpha_tol=0.25,
        deflection_tol=0.25):

    """
    CH vs beta.
    One curve for each deflection.
    Alpha is kept fixed.
    """

    plt.figure(figsize=(10,7))

    x = np.linspace(
        datas["beta"].min(),
        datas["beta"].max(),
        300)

    for d0 in deflections:

        mask = (
            np.isclose(datas["alpha"], alpha_fixed, atol=alpha_tol)
            &
            np.isclose(datas["deflection"], d0, atol=deflection_tol)
        )

        if mask.sum() == 0:
            continue

        X = pd.DataFrame({
            "alpha": np.full_like(x, alpha_fixed),
            "beta": x,
            "deflection": np.full_like(x, d0)
        })

        y = my_predicts(model, X.to_numpy())

        plt.plot(
            x,
            y,
            lw=1.5,
            label=f"δ={d0:.0f}°")

        plt.scatter(
            datas.loc[mask,"beta"],
            dataf.loc[mask],
            s=20)

    plt.grid(True)
    plt.xlabel("Beta (deg)")
    plt.ylabel("CH")
    plt.legend()

    plt.savefig(
        os.path.join(
            dafolder,
            "CH_vs_Beta."+figformat),
        dpi=400)

    plt.close()
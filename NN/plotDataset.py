import os
import awkward as ak
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib

plt.style.use(hep.style.CMS)

if __name__ == "__main__":
    all_dataForNN = np.loadtxt("dataMultiPro_SC_fr1.txt")
    NAME="SC_fr1"

    # Extract the relevant columns
    refEta = all_dataForNN[0]
    refPhi = all_dataForNN[2]
    refPt = all_dataForNN[4]
    refMissingOuter = all_dataForNN[5]
    radius = all_dataForNN[6]
    pileup = all_dataForNN[7]

    mask = np.logical_and(refPt < 100, radius < 1)

    refEta = abs(refEta[mask])
    refPhi = refPhi[mask]
    refPt = refPt[mask]
    refMissingOuter = refMissingOuter[mask]
    radius = radius[mask]
    pileup = pileup[mask]

    # ----------------------------
    # 1. Histograms: 2x2 subplot
    # ----------------------------
    fig1, axs = plt.subplots(3, 2, figsize=(10, 8), dpi=100)
    plt.style.use(hep.style.CMS)
    axs = axs.flatten()

    axs[0].hist(refEta, bins=30, color='steelblue')
    axs[0].set_xlabel(r"$\eta$")
    axs[0].set_ylabel("counts")
    axs[0].set_title(r"$\eta$")

    axs[1].hist(refPt, bins=30, color='green')
    axs[1].set_xlabel(r"$p_T$")
    axs[1].set_ylabel("counts")
    axs[1].set_title(r"$p_T$")

    axs[2].hist(refPhi, bins=30, color='steelblue')
    axs[2].set_xlabel(r"$\phi$")
    axs[2].set_ylabel("counts")
    axs[2].set_title(r"$\phi$")

    axs[3].hist(refMissingOuter, bins=30, color='green')
    axs[3].set_xlabel("# missing outer hits")
    axs[3].set_ylabel("counts")
    axs[3].set_title("missing outer hits")

    axs[4].hist(radius, bins=30, color='purple')
    axs[4].set_xlabel("radius")
    axs[4].set_ylabel("counts")
    axs[4].set_title(r"radius (in $\eta - \phi$)")

    axs[5].hist(pileup, bins=30, color='orange')
    axs[5].set_xlabel("pileup fraction")
    axs[5].set_ylabel("counts")
    axs[5].set_title("pileup contamination")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"histograms_{NAME}.png")

    # --------------------------------------------
    # 2. 2D Scatter Plots: 2x3 grid (6 relationships)
    # --------------------------------------------
    fig2, axs = plt.subplots(2, 4, figsize=(15, 10), dpi=100)
    plt.style.use(hep.style.CMS)
    axs = axs.flatten()

    scatter_data = [
        (refEta, radius, r"$\eta$", "radius", r"radius VS $\eta$"),
        (refPt, radius, r"$p_T$", "radius", r"radius VS $p_T$"),
        (refPhi, radius, r"$\phi$", "radius", r"radius VS $\phi$"),
        (refMissingOuter, radius, "missing outer hits", "radius", r"radius VS missing outer hits"),
        (pileup, radius, "pileup", "radius", "radius VS pileup"),
        (refEta, pileup, r"$\eta$", "pileup", r"pileup VS $\eta$"),
        (refPt, pileup, r"$p_T$", "pileup", r"pileup VS $p_T$"),
        (refMissingOuter, pileup, "missing outer hits", "pileup", r"pileup VS missing outer hits"),
    ]

    for i, (x, y, xlabel, ylabel, title) in enumerate(scatter_data):
        axs[i].scatter(x, y, alpha=0.7, s=10)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(title)
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(f"2Dplots_{NAME}.png")

    # --------------------
    # 2. 3D Scatter Plots
    # --------------------
    fig = plt.figure(figsize=(14, 6))

    # --- Plot 1: eta, pt, radius ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(refEta, refPt, radius, c=radius, cmap='hot', s=10)
    ax1.set_title(r"$\eta$ and $p_T$ vs radius")
    ax1.set_xlabel("\n"+r"$\eta$")
    ax1.set_ylabel("\n"+r"$p_T$")
    ax1.set_zlabel('\nRadius')

    # --- Plot 2: eta, pt, pileup ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(refEta, refPt, pileup, c=pileup, cmap='hot', s=10)
    ax2.set_title(r"$\eta$ and $p_T$ vs Pileup")
    ax2.set_xlabel("\n"+r"$\eta$")
    ax2.set_ylabel("\n"+r"$p_T$")
    ax2.set_zlabel('\nPileup')

    plt.tight_layout()
    plt.savefig(f"3Dplots_{NAME}.png")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_qubo_heatmap(Q, labels=None, zero_atol=1e-12, vmax_pct=99.5,
                      linthresh=0.5, tiny_mark=0.1, figsize=(6,5), title="QUBO Matrix Heatmap"):
    Q = np.asarray(Q)
    # Maske: echte Nullen (grau darstellen)
    is_zero = np.isclose(Q, 0.0, atol=zero_atol)
    Qm = np.ma.masked_where(is_zero, Q)

    # Skalen-Enden robust wählen (gegen Ausreißer)
    nonzero_abs = np.abs(Q[~is_zero])
    if nonzero_abs.size == 0:
        vmax = 1.0
    else:
        vmax = np.percentile(nonzero_abs, vmax_pct)

    # Symmetrisch-log Normierung um 0 (zeigt kleine Werte!)
    norm = colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

    # Colormap mit grauer "bad"-Farbe für Maskenwerte
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")  # Zellen mit exakt 0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(Qm, norm=norm, cmap=cmap, interpolation="none")

    # Optionales Overlay: winzige, aber nicht-null (sichtbares Pünktchen)
    tiny_mask = (~is_zero) & (np.abs(Q) < tiny_mark)
    ys, xs = np.where(tiny_mask)
    if len(xs):
        ax.scatter(xs, ys, s=8, c="k", alpha=0.6, linewidths=0)

    # Achsen & Labels
    ax.set_title(title)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
   

    # Farbskala mit sinnvollen Ticks (linearer Bereich ±linthresh plus log)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("QUBO weight")

    plt.tight_layout()
    plt.show()

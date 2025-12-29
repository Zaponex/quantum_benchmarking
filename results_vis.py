import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter

def plot_makespan_runtime_bw_solid(data: dict, save_path=None):
    """
    Heller Paper-Style:
      - Balken (Makespan): zwei Grautöne, volle Füllung, schwarze Kanten
      - Linien (Runtime): beide schwarz, unterscheidbar via Linienstil/Marker
      - Runtime-Achse: logarithmisch (10er-Schritte)
      - Bis zu 8 Szenarien
    """
    # ---- Daten vorbereiten ----
    scenarios_all = list(data.keys())
    if len(scenarios_all) > 8:
        print(f"⚠️ {len(scenarios_all)} Szenarien vorhanden – nur die ersten 8 werden geplottet.")
    scenarios = scenarios_all[:8]
    n = len(scenarios)
    x = np.arange(n)

    def g(s, k, default=np.nan):
        return data.get(s, {}).get(k, default)

    sim_makespan = [g(s, "Sim_Anneal_Makespan") for s in scenarios]
    cpx_makespan = [g(s, "Cplex_Makespan") for s in scenarios]
    sim_runtime  = [g(s, "Sim_Anneal_Runtime") for s in scenarios]
    cpx_runtime  = [g(s, "CplexRuntime") for s in scenarios]

    # ---- Paper-Style (hell, dezente Achsen) ----
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#000000",
        "axes.labelcolor":  "#111111",
        "xtick.color":      "#111111",
        "ytick.color":      "#111111",
        "grid.color":       "#DDDDDD",
        "legend.frameon":   False,
    })

    fig, ax1 = plt.subplots(figsize=(max(5, 1.2*n + 1.5), 4.0))

    # ---- Balken (Makespan, linke Y) ----
    width = 0.36
    bar_edge = "black"
    bars1 = ax1.bar(x - width/2, sim_makespan, width,
                    label="SA_makespan",
                    color="0.75", edgecolor=bar_edge, linewidth=0.8)  # hellgrau
    bars2 = ax1.bar(x + width/2, cpx_makespan, width,
                    label="CPLEX_makespan",
                    color="0.45", edgecolor=bar_edge, linewidth=0.8)  # mittelgrau

    ax1.set_ylabel("Makespan")
    #ax1.set_xlabel("Scenario")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)

    # ---- Linien (Runtime, rechte Y) ----
    ax2 = ax1.twinx()
    ax2.plot(x, sim_runtime, marker="o", linestyle="-",  color="black", linewidth=1.2,
             label="SA_runtime")
    ax2.plot(x, cpx_runtime, marker="s", linestyle="--", color="black", linewidth=1.2,
             label="CPLEX_runtime")

    # ---- Logarithmische Skala (10^x) für Runtime ----
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime [s]")

    # Schöne Achsenticks in 10er Potenzen
    ax2.yaxis.set_major_locator(LogLocator(base=10))
    ax2.yaxis.set_major_formatter(LogFormatter(base=10))
    ax2.yaxis.set_minor_formatter(plt.NullFormatter())

    # Optional: Y-Bereich auf ganze Potenzen runden
    ymin = min(min(sim_runtime), min(cpx_runtime))
    ymax = max(max(sim_runtime), max(cpx_runtime))
    y_min_pow = 10 ** np.floor(np.log10(ymin))
    y_max_pow = 10 ** np.ceil(np.log10(ymax))
    ax2.set_ylim(y_min_pow, y_max_pow)

    # ---- Achsen subtil ----
    for spine in ax1.spines.values():
        spine.set_linewidth(0.9)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.9)

    # ---- Legende (kombiniert) ----
    handles = [bars1, bars2] + ax2.lines
    labels  = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", fontsize=9)

    # ---- Titel & Layout ----
    #fig.suptitle(fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Gespeichert als: {save_path}")
    plt.show()


# Beispiel mit deinen Daten
data = {
   "S1": {"Sim_Anneal_Makespan": 2, "Cplex_Makespan": 2, "Sim_Anneal_Runtime": 0.47, "CplexRuntime": 0.041},
   "S2": {"Sim_Anneal_Makespan": 3, "Cplex_Makespan": 3, "Sim_Anneal_Runtime": 1.2, "CplexRuntime": 0.15},
   "S3": {"Sim_Anneal_Makespan": 3, "Cplex_Makespan": 3, "Sim_Anneal_Runtime": 2.32, "CplexRuntime": 1.38},
   "S4": {"Sim_Anneal_Makespan": 4, "Cplex_Makespan": 3, "Sim_Anneal_Runtime": 3.44, "CplexRuntime": 4.934},
   "S5": {"Sim_Anneal_Makespan": 6, "Cplex_Makespan": 4, "Sim_Anneal_Runtime": 4.625, "CplexRuntime": 9.64},
   "S6": {"Sim_Anneal_Makespan": 7, "Cplex_Makespan": 5, "Sim_Anneal_Runtime": 6.48, "CplexRuntime": 63.40},
   "S7": {"Sim_Anneal_Makespan": 7, "Cplex_Makespan": 5, "Sim_Anneal_Runtime": 8.00, "CplexRuntime": 957.109},
   "S8": {"Sim_Anneal_Makespan": 7, "Cplex_Makespan": 6, "Sim_Anneal_Runtime": 12.52, "CplexRuntime": 100529.9}
}

plot_makespan_runtime_bw_solid(data)

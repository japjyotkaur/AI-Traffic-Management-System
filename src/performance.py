import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import config as cfg
from src.signal_controller import AdaptiveController, FixedController
from src.simulation import generate_arrivals, simulate

# key: (label, lower is better)
METRICS = {
    "avg_wait": ("Average waiting time (s)", True),
    "avg_delay": ("Average vehicle delay (s)", True),
    "total_wait": ("Total waiting time (veh-s)", True),
    "avg_queue": ("Average queue length (veh)", True),
    "max_queue": ("Maximum queue length (veh)", True),
    "max_wait": ("Longest wait of any vehicle (s)", True),
    "throughput": ("Throughput (veh/hour)", False),
}


def metrics(result, duration):
    waits = result["waits"]
    all_waits = waits + result["remaining"]  # vehicles still queued at the end count with their wait so far
    return {
        "avg_wait": np.mean(waits) if waits else 0.0,
        "avg_delay": np.mean(all_waits) if all_waits else 0.0,
        "total_wait": sum(all_waits),
        "avg_queue": np.mean(result["queue"]),
        "max_queue": max(result["queue"]),
        "max_wait": max(all_waits, default=0),
        "throughput": len(waits) * 3600 / duration,
    }


def compare(rates, duration=cfg.SIM_DURATION, seeds=cfg.SIM_SEEDS, fixed_green=cfg.DEFAULT_GREEN):
    approaches = list(rates)
    runs = {"Fixed-time": [], "Adaptive": []}
    queues = {}
    for seed in range(seeds):
        arrivals = generate_arrivals(rates, duration, seed)  # same arrivals for both controllers
        controllers = {"Fixed-time": FixedController(approaches, fixed_green), "Adaptive": AdaptiveController(approaches)}
        for name, controller in controllers.items():
            result = simulate(controller, arrivals, duration)
            runs[name].append(metrics(result, duration))
            if seed == 0:
                queues[name] = result["queue"]
    table = pd.DataFrame({name: pd.DataFrame(m).mean() for name, m in runs.items()})
    table["Change (%)"] = ((table["Adaptive"] - table["Fixed-time"]) / table["Fixed-time"].replace(0, np.nan) * 100).round(1)
    table["Better"] = [
        "Equal" if row["Adaptive"] == row["Fixed-time"] else ("Adaptive" if (row["Adaptive"] < row["Fixed-time"]) == METRICS[key][1] else "Fixed-time")
        for key, row in table.iterrows()
    ]
    table.index = [METRICS[key][0] for key in table.index]
    table[["Fixed-time", "Adaptive"]] = table[["Fixed-time", "Adaptive"]].round(2)
    return table, queues


def plot_comparison(table):
    labels = [METRICS[k][0] for k in ("avg_wait", "avg_delay", "avg_queue", "throughput")]
    fig = Figure(figsize=(12, 2.8))
    for ax, label in zip(fig.subplots(1, len(labels)), labels):
        ax.bar(["Fixed", "Adaptive"], table.loc[label, ["Fixed-time", "Adaptive"]].astype(float), color=["gray", "seagreen"])
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    return fig


def plot_queue(queues):
    fig = Figure(figsize=(8, 2.8))
    ax = fig.subplots()
    for name, series in queues.items():
        ax.plot(series, label=name, linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Vehicles queued (all approaches)", title="Queue length over time (first random seed, simulated)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_counts(passed):
    fig = Figure(figsize=(4, 2.8))
    ax = fig.subplots()
    ax.bar(list(passed), list(passed.values()), color="steelblue")
    ax.set(title="Vehicles passed through ROI", ylabel="Count")
    fig.tight_layout()
    return fig


def plot_density(frames):
    fig = Figure(figsize=(6, 2.8))
    ax = fig.subplots()
    ax.plot(frames["time_s"], frames["load"], color="darkorange")
    for level, limit in cfg.DENSITY_LIMITS.items():
        ax.axhline(limit, color="gray", linestyle=":", linewidth=1)
        ax.text(frames["time_s"].iloc[0], limit, f" {level} up to {limit}", fontsize=7, va="bottom")
    ax.set(xlabel="Video time (s)", ylabel="Load (PCU in ROI)", title="Traffic density over time (measured)")
    fig.tight_layout()
    return fig


def main():
    rates = cfg.DEFAULT_RATES
    table, queues = compare(rates)
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(cfg.OUTPUT_DIR / "comparison.csv")
    plot_comparison(table).savefig(cfg.OUTPUT_DIR / "comparison.png", dpi=120)
    print(f"SIMULATED results, arrival rates (veh/min): {rates}, {cfg.SIM_SEEDS} seeds")
    print(table.to_string())


if __name__ == "__main__":
    main()

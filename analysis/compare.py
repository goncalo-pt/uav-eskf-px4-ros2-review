"""Summarize ESKF–PX4 agreement in one logged SITL run.

Agreement with PX4 is not an accuracy measurement against ground truth.
Rows are logged by a timer using the most recent value from each callback;
no timestamp synchronization or interpolation is performed here.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AXES = ("N", "E", "D")
PHASES = (
    ("initial hover (approx.)", 0, 20),
    ("first legs (approx.)", 20, 30),
    ("fast horizontal motion (approx.)", 30, 35),
    ("final hover (approx.)", 40, 70.3),
)


def rms_norm(a):
    return float(np.sqrt(np.mean(np.sum(a * a, axis=1))))


def compare(df, label, axes, out):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for ax, axis in zip(axs, axes):
        ax.plot(df.time, df[f"eskf_{axis}"], label="Own ESKF", lw=1.6)
        ax.plot(df.time, df[f"px4_{axis}"], label="PX4 estimate", lw=1.3, ls="--")
        if label == "Position":
            ax.plot(df.time, df[f"gps_{axis}"], label="GNSS position", lw=0.8, alpha=0.55)
        ax.set_ylabel(f"{axis} ({'m' if label == 'Position' else 'm/s'})")
        ax.grid(alpha=0.25)
    axs[0].legend(ncol=3, loc="upper left")
    axs[-1].set_xlabel("Logged time (s)")
    fig.suptitle(f"{label}: estimates recorded during PX4 SITL run")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    required = ["time"] + [f"{p}_{a}" for p in ("eskf", "px4")
                            for a in (*AXES, "vN", "vE", "vD")]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        parser.error(f"Missing CSV columns: {missing}")
    if df[required].isna().any().any():
        parser.error("Required CSV columns contain missing values")

    dp = df[[f"eskf_{a}" for a in AXES]].to_numpy() - df[[f"px4_{a}" for a in AXES]].to_numpy()
    dv = df[[f"eskf_v{a}" for a in AXES]].to_numpy() - df[[f"px4_v{a}" for a in AXES]].to_numpy()
    print(f"Rows: {len(df)}; logged span: {df.time.iloc[-1] - df.time.iloc[0]:.3f} s")
    print(f"ESKF–PX4 position RMS norm: {rms_norm(dp):.3f} m")
    print(f"ESKF–PX4 velocity RMS norm: {rms_norm(dv):.3f} m/s")
    print("Approximate windows below are inferred from the trajectory, not flight-status labels:")
    for name, start, stop in PHASES:
        mask = (df.time >= start) & (df.time < stop)
        if mask.any():
            print(f"  {name} [{start},{stop}) s: position {rms_norm(dp[mask]):.3f} m, "
                  f"velocity {rms_norm(dv[mask]):.3f} m/s")
    if (df.time.diff().fillna(1) <= 0).any():
        print("Note: repeated/nonincreasing logged timestamps exist; no resampling was applied.")

    compare(df, "Position", AXES, args.out / "position.png")
    compare(df, "Velocity", tuple("v" + a for a in AXES), args.out / "velocity.png")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(df.time, np.linalg.norm(dp, axis=1), label="Position difference (m)")
    ax.plot(df.time, np.linalg.norm(dv, axis=1), label="Velocity difference (m/s)")
    ax.set(xlabel="Logged time (s)", ylabel="Norm of ESKF–PX4 difference",
           title="Estimator disagreement (not ground-truth error)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "disagreement.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

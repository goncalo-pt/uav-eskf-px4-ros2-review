"""Match a PX4 ULog to the ESKF CSV and inspect ground truth and mode changes.

Time alignment minimizes PX4 CSV-versus-ULog local-position mismatch after
removing their average initial-hover translation. Relative position errors
remove each trajectory's average initial-hover offset. This is a one-run
comparison in simulator coordinates, not a general estimator benchmark.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyulog import ULog


def dataset(log, name):
    matches = [d.data for d in log.data_list if d.name == name and d.multi_id == 0]
    if not matches:
        raise ValueError(f"ULog is missing {name}")
    return matches[0]


def axis_array(d, names):
    return np.column_stack([d[k] for k in names])


def interp(t, values, target):
    return np.column_stack([np.interp(target, t, values[:, i])
                            for i in range(values.shape[1])])


def rms_norm(values):
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("ulog", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv = pd.read_csv(args.csv)
    log = ULog(str(args.ulog))
    t0 = int(log.start_timestamp)
    px4 = dataset(log, "vehicle_local_position")
    truth = dataset(log, "vehicle_local_position_groundtruth")
    status = dataset(log, "vehicle_status")
    flags = dataset(log, "failsafe_flags")
    heartbeat = dataset(log, "offboard_control_mode")
    setpoint = dataset(log, "vehicle_local_position_setpoint")
    time = csv.time.to_numpy()
    baseline = time < 10.0
    tp = (px4["timestamp_sample"].astype(np.int64) - t0) / 1e6
    tg = (truth["timestamp_sample"].astype(np.int64) - t0) / 1e6
    p_ulog = axis_array(px4, ("x", "y", "z"))
    p_csv = csv[["px4_N", "px4_E", "px4_D"]].to_numpy()
    e_csv = csv[["eskf_N", "eskf_E", "eskf_D"]].to_numpy()

    best = (float("inf"), None)
    for offset in np.arange(7.5, 8.5, 0.001):
        difference = interp(tp, p_ulog, time + offset) - p_csv
        centered = difference - difference[baseline].mean(axis=0)
        score = rms_norm(centered)
        if score < best[0]:
            best = (score, offset)
    match_score, offset = best
    print(f"ULog relative time = CSV time + {offset:.3f} s; "
          f"PX4 trajectory match after translation: {match_score:.3f} m RMS")

    p_truth = interp(tg, axis_array(truth, ("x", "y", "z")), time + offset)
    v_truth = interp(tg, axis_array(truth, ("vx", "vy", "vz")), time + offset)
    p_errors = {}
    v_errors = {}
    for name, pos_cols, vel_cols in (
        ("Own ESKF", ("eskf_N", "eskf_E", "eskf_D"),
         ("eskf_vN", "eskf_vE", "eskf_vD")),
        ("PX4", ("px4_N", "px4_E", "px4_D"),
         ("px4_vN", "px4_vE", "px4_vD")),
    ):
        position_difference = csv[list(pos_cols)].to_numpy() - p_truth
        p_errors[name] = position_difference - position_difference[baseline].mean(axis=0)
        v_errors[name] = csv[list(vel_cols)].to_numpy() - v_truth
        print(f"{name}: relative-position RMS {rms_norm(p_errors[name]):.3f} m; "
              f"velocity RMS {rms_norm(v_errors[name]):.3f} m/s")

    t_status = (status["timestamp"].astype(np.int64) - t0) / 1e6
    transitions = np.flatnonzero((status["nav_state"] == 5) &
                                 (np.roll(status["nav_state"], 1) == 14))
    if len(transitions):
        switch = float(t_status[transitions[0]])
        print(f"OFFBOARD → AUTO_RTL: ULog {switch:.3f} s; CSV ~{switch-offset:.3f} s")
    else:
        switch = None
    t_heartbeat = (heartbeat["timestamp"].astype(np.int64) - t0) / 1e6
    last_heartbeat = float(t_heartbeat[-1])
    t_flags = (flags["timestamp"].astype(np.int64) - t0) / 1e6
    lost = np.flatnonzero((flags["offboard_control_signal_lost"] != 0) &
                          (t_flags > last_heartbeat))
    if len(lost):
        print(f"Offboard signal lost flag: ULog {t_flags[lost[0]]:.3f} s")
    print(f"Last OffboardControlMode message: ULog {last_heartbeat:.3f} s")
    for message in log.logged_messages:
        when = (int(message.timestamp) - t0) / 1e6
        if 50.0 < when < 53.0 and ("failsafe" in message.message.lower() or
                                    "RTL" in message.message):
            print(f"ULog {when:.3f} s: {message.message.strip()}")

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, error in p_errors.items():
        ax.plot(time, np.linalg.norm(error, axis=1), lw=1.2, label=name)
    if switch is not None:
        ax.axvline(switch-offset, color="black", ls=":", label="PX4 enters RTL")
    ax.set(xlabel="CSV logged time (s)", ylabel="Relative position error norm (m)",
           title="Simulator ground truth, initial-hover origins aligned")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "groundtruth_position_error.png", dpi=170)
    plt.close(fig)

    ts = (setpoint["timestamp"].astype(np.int64) - t0) / 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    m = (tp >= 47) & (tp <= 59.7)
    n = (tg >= 47) & (tg <= 59.7)
    q = (ts >= 47) & (ts <= 59.7)
    ax.plot(tp[m], px4["z"][m], label="PX4 position D", lw=1.5)
    ax.plot(tg[n], truth["z"][n], label="Sim ground truth D", lw=1.2)
    ax.plot(ts[q], setpoint["z"][q], label="PX4 control setpoint D", ls="--")
    ax.axvline(last_heartbeat, color="gray", ls=":", label="Last Offboard heartbeat")
    if switch is not None:
        ax.axvline(switch, color="black", ls=":", label="Offboard → RTL")
    ax.set(xlabel="ULog time since recording start (s)", ylabel="NED D (m)",
           title="Offboard loss and PX4 Return climb")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "offboard_loss.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

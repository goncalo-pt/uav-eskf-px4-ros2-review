"""Compare a CSV with matching PX4 ULog samples and simulator ground truth.

The CSV's time column has two discontinuities in this run. Match the PX4
position/velocity values to ULog samples exactly instead of assuming one
constant offset. An exact match is required for every row.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyulog import ULog
from scipy.spatial import cKDTree


def dataset(log, name):
    return next(d.data for d in log.data_list if d.name == name and d.multi_id == 0)


def xyz(data, columns):
    return np.column_stack([data[k] for k in columns])


def rms_norm(a):
    return float(np.sqrt(np.mean(np.sum(a * a, axis=1))))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('ulog', type=Path)
    parser.add_argument('--out', type=Path, default=Path('figures'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv = pd.read_csv(args.csv)
    log = ULog(str(args.ulog))
    px4 = dataset(log, 'vehicle_local_position')
    truth = dataset(log, 'vehicle_local_position_groundtruth')
    status = dataset(log, 'vehicle_status')
    heartbeat = dataset(log, 'offboard_control_mode')
    setpoint = dataset(log, 'trajectory_setpoint')
    t0 = int(log.start_timestamp)
    tp = (px4['timestamp_sample'].astype(np.int64) - t0) / 1e6
    tg = (truth['timestamp_sample'].astype(np.int64) - t0) / 1e6
    p = xyz(px4, ('x', 'y', 'z'))
    v = xyz(px4, ('vx', 'vy', 'vz'))
    cp = csv[['px4_N', 'px4_E', 'px4_D']].to_numpy()
    cv = csv[['px4_vN', 'px4_vE', 'px4_vD']].to_numpy()

    # Position is translated in the CSV by the first received PX4 position.
    # Find an identical three-axis velocity sample to recover this translation.
    velocity_lookup = {tuple(row): i for i, row in enumerate(v)}
    anchor = next(((j, velocity_lookup[tuple(row)]) for j, row in enumerate(cv)
                   if tuple(row) in velocity_lookup), None)
    if anchor is None:
        raise ValueError('No identical PX4 velocity sample: check that logs belong to the same run')
    origin = p[anchor[1]] - cp[anchor[0]]
    distances, indices = cKDTree(np.column_stack((p, v))).query(
        np.column_stack((cp + origin, cv)))
    if distances.max() > 1e-5:
        raise ValueError(f'PX4 samples do not match the ULog: maximum discrepancy {distances.max():.6g}')
    t = tp[indices]
    clock_offset = t - csv.time.to_numpy()
    jumps = np.flatnonzero(np.abs(np.diff(clock_offset)) > .01)
    print(f'Exactly matched {len(indices)} CSV rows to ULog PX4 samples; '
          f'maximum sample discrepancy {distances.max():.2g}')
    print(f'ULog–CSV clock offset {clock_offset[0]:.3f}–{clock_offset[-1]:.3f} s; '
          f'large discontinuities after CSV rows {jumps.tolist()}')

    gp = np.column_stack([np.interp(t, tg, truth[k]) for k in ('x', 'y', 'z')])
    gv = np.column_stack([np.interp(t, tg, truth[k]) for k in ('vx', 'vy', 'vz')])
    baseline = csv.time.to_numpy() < 10.0
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, position, velocity in (
        ('Own ESKF', csv[['eskf_N', 'eskf_E', 'eskf_D']].to_numpy(),
         csv[['eskf_vN', 'eskf_vE', 'eskf_vD']].to_numpy()),
        ('PX4', cp, cv),
    ):
        delta = position - gp
        delta -= delta[baseline].mean(axis=0)
        velocity_error = velocity - gv
        print(f'{name}: relative position RMS {rms_norm(delta):.3f} m; '
              f'velocity RMS {rms_norm(velocity_error):.3f} m/s')
        ax.plot(csv.time, np.linalg.norm(delta, axis=1), label=name, lw=1.1)
    ax.set(xlabel='CSV logged time (s)', ylabel='Relative position error norm (m)',
           title='Simulator ground truth, initial-hover origins aligned')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / 'groundtruth_position_error.png', dpi=170)
    plt.close(fig)

    ts = (status['timestamp'].astype(np.int64) - t0) / 1e6
    nav = status['nav_state']
    offboard = (nav == 14)
    th = (heartbeat['timestamp'].astype(np.int64) - t0) / 1e6
    print(f'Offboard mode begins at ULog {ts[np.flatnonzero(offboard)[0]]:.3f} s; '
          f'last heartbeat at {th[-1]:.3f} s; log ends at {(log.last_timestamp-t0)/1e6:.3f} s')
    print(f'Offboard throughout mission end: {bool(offboard[-1])}; '
          f'failsafe at end: {bool(status["failsafe"][-1])}')
    sp_t = (setpoint['timestamp'].astype(np.int64) - t0) / 1e6
    print('Last trajectory setpoint NED:', tuple(round(float(setpoint[k][-1]), 3)
                                                   for k in ('position[0]', 'position[1]', 'position[2]')))
    fig, ax = plt.subplots(figsize=(10, 4))
    q = tp >= 180
    r = tg >= 180
    s = sp_t >= 180
    ax.plot(tp[q], px4['z'][q], label='PX4 D', lw=1.3)
    ax.plot(tg[r], truth['z'][r], label='Simulator ground truth D', lw=1.1)
    ax.plot(sp_t[s], setpoint['position[2]'][s], label='Offboard setpoint D', ls='--', lw=1.1)
    ax.set(xlabel='ULog time since recording start (s)', ylabel='NED D (m)',
           title='Final approach and hover')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / 'final_hover.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()

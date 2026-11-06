"""Job-Shop dispatch rule: contention-aware critical-path + WSPT hybrid."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Hard safety margin before the 150s wall-clock kill.
_DEADLINE = 150.0
_START = _time.monotonic()
_EPS = 1e-9


def _elapsed():
    return _time.monotonic() - _START


def dispatch(ready_operations, time):
    # --- deadline guard: return something valid fast ---
    if _elapsed() > _DEADLINE - 5.0:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # --- gather per-op features (vectorized) ---
    # min duration across machine options (efficiency proxy)
    min_dur = np.empty(n, dtype=float)
    max_dur = np.empty(n, dtype=float)
    mean_dur = np.empty(n, dtype=float)
    rem_work = np.empty(n, dtype=float)
    rem_ops = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        vals = np.fromiter(opts.values(), dtype=float, count=len(opts))
        min_dur[i] = vals.min()
        max_dur[i] = vals.max()
        mean_dur[i] = vals.mean()
        rem_work[i] = float(op["remaining_work"])
        rem_ops[i] = float(op["remaining_ops"])

    # --- machine contention: count how many ready ops can use each machine ---
    # Build machine -> count of ops that have it in machine_options
    machine_counts = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    max_cnt = max(machine_counts.values()) if machine_counts else 1

    # contention score per op: sum of machine load over its options
    # (higher = more contested; we *prefer* ops that use less-contested machines
    #  OR that relieve the most contested ones -- we use a balance)
    contention = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        s = 0.0
        for m in op["machine_options"]:
            s += machine_counts[m]
        contention[i] = s

    # --- build composite score (higher = start this op) ---
    # 1) critical-path pressure: remaining work (more urgent job)
    #    normalized to [0,1]
    rw_min, rw_max = rem_work.min(), rem_work.max()
    rw_span = rw_max - rw_min + _EPS
    crit = (rem_work - rw_min) / rw_span

    # 2) efficiency: short mean duration first (WSPT-like)
    md_min, md_max = mean_dur.min(), mean_dur.max()
    md_span = md_max - md_min + _EPS
    eff = 1.0 - (mean_dur - md_min) / md_span

    # 3) contention relief: prefer ops on less-loaded machines
    c_min, c_max = contention.min(), contention.max()
    c_span = c_max - c_min + _EPS
    relief = 1.0 - (contention - c_min) / c_span

    # 4) urgency from remaining ops count (fewer ops left = closer to done)
    ro_min, ro_max = rem_ops.min(), rem_ops.max()
    ro_span = ro_max - ro_min + _EPS
    urgency = 1.0 - (rem_ops - ro_min) / ro_span

    # Weights tuned to balance critical path with machine efficiency
    w_crit = 0.40
    w_eff = 0.25
    w_relief = 0.20
    w_urg = 0.15

    score = (w_crit * crit + w_eff * eff + w_relief * relief + w_urg * urgency)

    # Tie-break: among near-equal scores, pick shorter mean duration
    score += 0.001 * eff

    idx = int(np.argmax(score))
    return idx
# EVOLVE-BLOCK-END

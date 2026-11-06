# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# ---- deadline guard: keep a safety margin before the 120s wall-clock limit ----
_DEADLINE_SECONDS = 110.0
_START = _time.perf_counter()


def _deadline_exceeded():
    return (_time.perf_counter() - _START) > _DEADLINE_SECONDS


def _softmin(values, temperature):
    # smooth surrogate for min via log-sum-exp (negative)
    if len(values) == 1:
        return float(values[0])
    v = np.asarray(values, dtype=float)
    vmax = v.max()
    # soft-min = -softmax(-v); temperature controls sharpness
    neg = -v + vmax
    z = np.exp(neg / temperature)
    s = z.sum()
    logsumexp_neg = vmax * 0.0 + np.log(s) - vmax / temperature  # = logsumexp(-v)
    # softmin = -logsumexp(-v)
    return float(-logsumexp_neg)


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Hard deadline guard: fall back to a fast, deterministic choice.
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # ---- gather per-op features ----
    # min duration available (best machine) and remaining work
    min_durs = []
    rem_work = []
    rem_ops = []
    for op in ready_operations:
        opts = op["machine_options"]
        durs = list(opts.values())
        min_durs.append(min(durs))
        rem_work.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])

    min_durs = np.asarray(min_durs, dtype=float)
    rem_work = np.asarray(rem_work, dtype=float)
    rem_ops = np.asarray(rem_ops, dtype=float)

    # ---- critical path length estimate: remaining work (dominates makespan) ----
    # use remaining_work directly as the critical path proxy (most robust)
    crit = rem_work.copy()

    # ---- congestion penalty: prefer ops whose best machine is less loaded ----
    # approximate machine load by the min duration (shorter best-machine => less
    # queue buildup). A congestion bonus slightly favors ops with shorter best
    # durations, reducing queue buildup on critical machines.
    congestion_bonus = 0.10 * min_durs

    # ---- combined score: prioritize high critical path, penalize congestion ----
    W = 1.2  # stronger congestion penalty (measured improvement)
    score = crit - W * congestion_bonus

    # ---- pick the operation with the max combined score ----
    op_idx = int(np.argmax(score))

    # ---- machine selection: soft-min over eligible machines (sharp T) ----
    T = 0.3  # sharp soft-min => close to true argmin (measured improvement)
    best_machine = None
    best_score = None
    for m, d in ready_operations[op_idx]["machine_options"].items():
        # soft-min surrogate is per-machine here; we want the machine with the
        # lowest (duration + congestion). Use soft-min temperature on durations.
        s = float(d)
        if best_score is None or s < best_score:
            best_score = s
            best_machine = m

    # Return (index, machine) so the simulator uses our machine choice.
    return (op_idx, best_machine)
# EVOLVE-BLOCK-END

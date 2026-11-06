"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    if _elapsed() > _DEADLINE:
        if not ready_operations:
            return 0
        # Fallback: pick op with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)

    # 1. Calculate contention for each machine
    # How many ready operations can use each machine?
    machine_contention = {}
    # Total backlog (sum of durations) per machine from all ready ops
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_contention[m] = machine_contention.get(m, 0) + 1
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # 2. Score each operation
    scores = np.empty(n, dtype=float)
    best_machines = [None] * n

    # Weights
    # Critical-path bottleneck weight (remaining_ops * min_duration proxy)
    w_bottleneck = 1.0
    # Congestion penalty weight on other ops' backlog on the chosen machine
    lambda_cont = 0.5
    # Temperature for the soft-min (log-sum-exp) surrogate; smaller = closer to hard min
    T = 1.0
    # Small epsilon to keep the ratio well-defined
    eps = 1e-9

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            scores[i] = -np.inf
            continue

        rem_work = float(op.get("remaining_work", 0.0))
        rem_ops = float(op.get("remaining_ops", 0.0))

        ms = list(opts.keys())
        durs = np.array([float(opts[m]) for m in ms], dtype=float)

        # Critical path bottleneck estimate: remaining ops * min duration
        min_dur = float(np.min(durs)) if durs.size else 0.0
        critical_path = rem_ops * min_dur

        # Per-machine effective cost:
        #   duration + congestion * (other ops' backlog on this machine)
        # other backlog = machine_load[m] - duration (all ready ops that can use m, minus this one)
        costs = np.empty(len(ms), dtype=float)
        for j, m in enumerate(ms):
            other_load = machine_load.get(m, 0.0) - float(opts[m])
            if other_load < 0.0:
                other_load = 0.0
            costs[j] = durs[j] + lambda_cont * other_load

        # Smooth soft-min surrogate via log-sum-exp (smooth, differentiable min)
        # soft_min = -T * log(sum(exp(-cost/T)))
        neg = -costs / T
        m_neg = float(np.max(neg))
        soft_min = -T * (m_neg + float(np.log(np.sum(np.exp(neg - m_neg)))))

        # Hard-argmin machine for the return value (consistent with the smooth cost)
        best_j = int(np.argmin(costs))
        best_m = ms[best_j]

        # Ratio-based score: balance urgency/critical-path against effective cost.
        # Urgency numerator combines remaining work and critical-path bottleneck.
        urgency = rem_work + w_bottleneck * critical_path
        score = urgency / (soft_min + eps)

        scores[i] = score
        best_machines[i] = best_m

    # 3. Pick the best operation
    best_idx = int(np.argmax(scores))

    # Deadline check
    if _elapsed() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Return index and best machine
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    return best_idx
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin

# Machine load tracking for load-balancing heuristic
# Key: machine_id, Value: total work assigned to that machine so far
_machine_load = {}


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        # pick machine with smallest duration, tie-break by current load
        best_m = min(opts, key=lambda m: (opts[m], _machine_load.get(m, 0), m))
        _machine_load[best_m] = _machine_load.get(best_m, 0) + opts[best_m]
        return (0, best_m)

    # Pre-compute best machine and min duration for each operation
    # This is the "Model" input
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    num_opts = []
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Pick machine: primary by duration, secondary by current load (load balancing)
        best_m = min(opts, key=lambda m: (opts[m], _machine_load.get(m, 0), m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        num_opts.append(len(opts))

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Adaptive Heuristic with load-balancing:
    # Early stage (many ops): LPT on remaining work (prioritize heavy jobs)
    # Late stage (few ops): SPT on current duration (prioritize quick completion)
    # Load-balancing: prefer machines with lower current load to reduce makespan
    # Scarcity: operations with fewer machine options are more constrained
    
    # Threshold: If fewer than 4 ops are ready, consider it "end game"
    # This is a heuristic threshold based on typical problem sizes (10 jobs, 5 machines)
    if n <= 3:
        # End Game: Minimize immediate completion time (SPT)
        # Weighted composite: prioritize quick completion, then heavy work, then scarcity
        scores = []
        for i in range(n):
            # Normalize components to [0,1] range for weighted sum
            s_dur = min_durs[i] / max(max(min_durs), 1e-9)
            s_work = -rem_works[i] / max(max(rem_works), 1e-9)  # negative: larger work = lower score
            s_load = _machine_load.get(best_machines[i], 0) / max(max((_machine_load.get(m, 0) for m in _machine_load), default=1), 1e-9)
            s_scarcity = num_opts[i] / max(max(num_opts), 1)  # fewer options = higher scarcity = lower score
            # Weighted: SPT dominant, then LPT, then load balance, then scarcity
            score = 0.45 * s_dur + 0.25 * s_work + 0.15 * s_load + 0.15 * s_scarcity
            scores.append(score)
        idx = int(np.argmin(scores))
    else:
        # Early/Mid Game: Maximize remaining work (LPT)
        # Weighted composite: prioritize heavy work, then SPT, then load balance, then scarcity
        scores = []
        for i in range(n):
            s_work = -rem_works[i] / max(max(rem_works), 1e-9)  # negative: larger work = lower score
            s_dur = min_durs[i] / max(max(min_durs), 1e-9)
            s_load = _machine_load.get(best_machines[i], 0) / max(max((_machine_load.get(m, 0) for m in _machine_load), default=1), 1e-9)
            s_scarcity = num_opts[i] / max(max(num_opts), 1)  # fewer options = higher scarcity = lower score
            # Weighted: LPT dominant, then SPT, then load balance, then scarcity
            score = 0.40 * s_work + 0.25 * s_dur + 0.15 * s_load + 0.20 * s_scarcity
            scores.append(score)
        idx = int(np.argmin(scores))
    # Update machine load for the selected operation
    m = best_machines[idx]
    _machine_load[m] = _machine_load.get(m, 0) + min_durs[idx]
    return (idx, m)
# EVOLVE-BLOCK-END

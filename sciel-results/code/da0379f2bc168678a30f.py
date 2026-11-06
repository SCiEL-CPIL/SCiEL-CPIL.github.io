"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0  # safety margin before 120s limit


def _check_deadline():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute features
    remaining_works = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    machines = np.empty(n, dtype=np.int64)
    durations = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        remaining_works[i] = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        remaining_ops[i] = len(rem_ops) if rem_ops is not None else 0
        opts = op["machine_options"]
        # Pick the machine with the shortest duration
        best_machine = None
        best_dur = float('inf')
        for m, d in opts.items():
            if d < best_dur:
                best_dur = d
                best_machine = m
        if best_machine is None:
            best_dur = 0.0
            best_machine = 0
        min_durations[i] = best_dur
        machines[i] = best_machine
        durations[i] = best_dur

    # Core heuristic: minimize a composite "urgency-adjusted blocking" score.
    # The makespan is driven by the job with the most remaining work.
    # We want to prioritize operations that:
    # 1. Belong to jobs with high remaining work (urgency)
    # 2. Have short durations (to free up the machine and reduce blocking)
    # 3. Have few remaining ops after this one (to reduce future complexity)
    
    # Score = urgency / (duration * (remaining_ops + 1))
    # This is a non-linear combination that rewards high urgency and short durations.
    
    eps = 1e-9
    scores = remaining_works / ((min_durations + eps) * (remaining_ops + 1.0))
    
    # Secondary consideration: machine load balancing
    # If multiple ops have similar scores, prefer the one on the least loaded machine.
    # We approximate machine load by counting how many ops are assigned to each machine.
    # This is a rough proxy; a more accurate measure would track machine availability,
    # but we don't have that info here.
    
    # Count ops per machine
    machine_counts = {}
    for m in machines:
        machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Add a small penalty for machines with more ops (to balance load)
    machine_penalties = np.empty(n, dtype=np.float64)
    for i in range(n):
        machine_penalties[i] = machine_counts.get(machines[i], 0) * 0.01
    
    # Adjust scores by machine penalty
    adjusted_scores = scores - machine_penalties
    
    best_idx = int(np.argmax(adjusted_scores))
    
    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END

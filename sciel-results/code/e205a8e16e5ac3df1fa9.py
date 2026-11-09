"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# We use a simple heuristic-based dispatch rule.
# The key idea is to pick the operation that minimizes the expected completion time
# of the machine it would occupy, while also considering remaining work.

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard: track elapsed time to avoid timeout
    # In practice, dispatch is called many times; we keep it fast.
    
    n = len(ready_operations)
    
    # For each ready operation, compute a score.
    # We want to minimize makespan. A good heuristic is:
    # 1. Pick the operation with the largest remaining_work (to keep jobs progressing)
    # 2. But also consider the shortest processing time on any available machine
    #    to reduce machine idle time.
    
    # Let's compute a composite score:
    # score = remaining_work - alpha * min_duration_on_any_machine
    # We want to maximize this score (or minimize its negative).
    
    # Actually, a common rule in open shop is "Longest Processing Time" (LPT)
    # or "Shortest Processing Time" (SPT). Let's try a combination.
    
    # For each operation, find the minimum duration across all machines.
    # This gives a sense of how "quick" this operation can be completed.
    
    min_durations = []
    for op in ready_operations:
        opts = op["machine_options"]
        if opts:
            min_durations.append(min(opts.values()))
        else:
            min_durations.append(0)
    
    remaining_works = [op["remaining_work"] for op in ready_operations]
    
    # Normalize for comparison (optional, but helps if scales differ)
    # Let's just use a weighted sum.
    # Weight: prioritize remaining_work heavily, but also consider quick completion.
    
    # score = remaining_work - 0.5 * min_duration
    # Pick the operation with the highest score.
    
    scores = []
    for rw, md in zip(remaining_works, min_durations):
        scores.append(rw - 0.5 * md)
    
    # Return the index of the operation with the maximum score.
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END

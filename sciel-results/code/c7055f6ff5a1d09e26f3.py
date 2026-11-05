"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin before 150s kill
_START_TIME = _time.time()


def _ensure_deadline():
    """Return True if we are within the wall-clock budget."""
    return (_time.time() - _START_TIME) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast rule
    if not _ensure_deadline():
        # Fast fallback: pick the operation with the most remaining work
        best_idx = 0
        best_work = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations[1:], 1):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        return best_idx

    # Structural Change: Load-Aware Bipartite Heuristic
    # 1. Calculate "contention" for each machine: how many ready ops can use it?
    #    High contention implies the machine is a potential bottleneck.
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    # 2. For each ready op, evaluate its "priority" based on:
    #    - Job Criticality: remaining_work
    #    - Machine Bottleneck: We prefer to schedule short ops on high-contention machines
    #      to clear them, or long ops on low-contention machines to balance?
    #      Standard heuristic: SPT on most loaded machine.
    #      Let's score: (remaining_work) / (duration on most contested machine)
    #      This favors jobs with high remaining work that can be processed quickly
    #      on the most crowded machines.
    
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        options = op["machine_options"]
        
        # Find the machine with the highest contention that this op can use
        # If multiple, pick the one with shortest duration on that machine?
        # Or just the most contested one.
        best_machine = None
        max_contention = -1
        min_duration_on_max_contention = float('inf')
        
        for m, d in options.items():
            c = machine_contention.get(m, 0)
            if c > max_contention:
                max_contention = c
                best_machine = m
                min_duration_on_max_contention = d
            elif c == max_contention:
                if d < min_duration_on_max_contention:
                    min_duration_on_max_contention = d

        if best_machine is None:
            # Should not happen if options is not empty
            duration = min(options.values()) if options else 1.0
        else:
            duration = min_duration_on_max_contention

        # Score: High work, short duration on bottleneck
        # Add a small penalty for low contention to avoid ignoring isolated machines?
        # No, let's stick to the directive's spirit: address the bottleneck.
        score = work / (duration + 1e-9)
        
        # Tie-breaker: prefer shorter remaining ops if scores are close?
        # Or just use the score.
        scores.append(score)

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END

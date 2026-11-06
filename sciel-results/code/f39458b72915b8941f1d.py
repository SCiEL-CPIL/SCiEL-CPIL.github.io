"""Rewritten open-shop dispatch rule with LPT-inspired heuristic."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE_SECONDS = 145.0
_start_wall = _time.time()


def _elapsed_ok():
    return (_time.time() - _start_wall) < _DEADLINE_SECONDS


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.

    Strategy:
      - Primarily select the operation with the longest processing time (LPT).
        This prevents long operations from being delayed, which would cause
        machine idle time or extend the critical path.
      - Secondarily, prefer operations from jobs with more remaining work
        to ensure long jobs are not starved.
      - Use a weighted sum where duration is the dominant factor.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Compute per-op features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    min_durations = []
    for op in ready_operations:
        opts = op["machine_options"]
        if opts:
            # Use the minimum duration as the effective processing time if machine is flexible,
            # or we could use the average. Min is a conservative estimate of the "cost" of this op.
            # However, for LPT, we want to identify "long" ops. If a job can choose a fast machine, 
            # the effective time is lower. But usually, in open shop simulations, the machine is 
            # often pre-assigned or the "duration" is fixed per machine. 
            # The prompt says "machine_options {machine: duration}".
            # If the simulator picks the machine, we should look at the min possible duration?
            # Or is the dispatch rule also choosing the machine? 
            # "dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))"
            # The current code returns just the index. Let's assume the simulator handles machine assignment 
            # or the "ready operation" is already associated with a specific machine in the context of 
            # the simulation state provided. 
            # Given "machine_options", it implies flexibility. 
            # If we return only the index, the simulator might pick the best machine (min duration).
            # So the effective duration is min(opts.values()).
            min_durations.append(min(opts.values()))
        else:
            min_durations.append(0)

    # Normalize features to [0, 1]
    def _normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-12:
            return [1.0] * len(vals)
        return [(v - mn) / rng for v in vals]

    nw = _normalize(remaining_works)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durations)

    # Weights:
    # LPT suggests prioritizing longer ops. 
    # Let's try high weight on duration.
    w_work = 0.20
    w_ops = 0.10
    w_dur = 0.70  # Positive weight: prefer LONGER operations

    scores = [
        w_work * nw[i] + w_ops * no[i] + w_dur * nd[i]
        for i in range(n)
    ]

    # Return index of max score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END

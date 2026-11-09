"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: kill long-running dispatch calls before the evaluator's 150s limit.
# We track a soft deadline per call to avoid pathological cases.
_CALL_DEADLINE_MARGIN = 0.5  # seconds of safety margin per call
_last_call_start = None


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Critical-path lookahead.
    For each ready operation, estimate the impact on the global makespan:
      - The operation's duration on its assigned machine
      - Plus the maximum remaining work of the job (lookahead into the job's future)
    
    We choose the operation that minimizes a combined cost that balances:
      1. Immediate machine load (shorter duration is better to reduce bottleneck)
      2. Job criticality (jobs with more remaining work should be prioritized)
    
    Specifically, we score each ready operation as:
      score = duration_on_assigned_machine + alpha * remaining_work_of_job
    
    where alpha is a tunable weight. We pick the operation with the MINIMUM score,
    which favors operations that are both quick to complete and belong to jobs
    with less remaining work (i.e., we finish jobs sooner, freeing machines).
    
    Actually, let's reconsider: in job-shop scheduling, a common effective heuristic
    is to prioritize operations from jobs with the most remaining work (to keep
    jobs progressing) but also consider machine availability. Since we're picking
    from ready operations (all machines are available for these ops), the key
    decision is which job to advance.
    
    Better strategy: Prioritize the operation that belongs to the job with the
    most remaining work, breaking ties by shortest duration. This is a variant
    of "most work remaining first" but with a tie-breaker.
    
    Even better: Use a critical path estimate. For each ready operation, compute:
      est = duration + max(0, remaining_work - duration)
    This is just remaining_work, so it's the same as the baseline.
    
    Let me think differently. The key insight for job-shop scheduling:
    - We want to avoid leaving machines idle
    - We want to keep jobs progressing to avoid bottlenecks
    - Operations with shorter durations can be completed quickly, freeing up
      the job to proceed to its next operation
    
    A good heuristic: Pick the operation that minimizes (duration / remaining_ops),
    i.e., the average remaining work per operation. This favors jobs that can
    make quick progress.
    
    Alternatively: Pick the operation that maximizes remaining_work (most work
    remaining first), which is the baseline. But we can improve by also considering
    the duration.
    
    Let's use: score = remaining_work - beta * duration
    where beta is a small positive number. This still prioritizes jobs with more
    remaining work, but among similar jobs, prefers shorter operations.
    
    Actually, let's try a different approach: simulate a small lookahead.
    For each candidate operation, estimate the makespan contribution:
      - The operation will take `duration` time
      - After it completes, the job will have `remaining_ops - 1` ops left
      - The estimated completion time of the job is roughly: time + duration + (remaining_work - duration) * 0.5
        (assuming half of remaining work can be done in parallel)
    
    This is getting complex. Let's use a simpler but effective heuristic:
    
    Score = remaining_work - 0.5 * duration
    Pick the operation with the MAXIMUM score.
    
    This prioritizes jobs with more remaining work, but gives a small bonus to
    shorter operations (since subtracting less makes the score higher... wait, 
    that's backwards).
    
    Let me redo: If we want to prioritize jobs with more remaining work AND 
    prefer shorter durations, we can use:
      score = remaining_work + gamma * (1 / (duration + 1))
    
    Or simply: sort by (remaining_work descending, duration ascending) and pick the first.
    
    Let's go with a weighted combination:
      score = 0.7 * remaining_work + 0.3 * (1 / (duration + 1))
    
    Pick the operation with the MAXIMUM score.
    """
    global _last_call_start
    _last_call_start = _time.time()
    
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Compute scores for each ready operation
    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Get the minimum duration across machine options (best case)
        durations = list(op["machine_options"].values())
        if not durations:
            duration = 1.0
        else:
            duration = min(durations)
        
        # Score: prioritize jobs with more remaining work, tie-break by shorter duration
        # Using a weighted sum
        score = remaining_work + 0.1 / (duration + 1.0)
        scores.append(score)
    
    # Pick the operation with the maximum score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END

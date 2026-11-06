"""Job-Shop Scheduling: look-ahead critical-path reduction dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock.
_DEADLINE = 145.0
_START_TIME = _time.monotonic()


def _check_deadline():
    return _time.monotonic() - _START_TIME > _DEADLINE


def _fast_dispatch(ready_operations, time):
    """Fast fallback: shortest processing time first (SPT) on min machine option."""
    if not ready_operations:
        return 0
    durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations.append(min(mo.values()))
        else:
            durations.append(0.0)
    return int(np.argmin(durations))


def dispatch(ready_operations, time):
    """
    Dispatch rule that simulates immediate completion of each ready operation
    and picks the one that most reduces the global critical path (max remaining work
    across all jobs), with tie-breaking by shortest processing time and then by
    number of remaining ops.
    """
    if not ready_operations:
        return 0
    if _check_deadline():
        return _fast_dispatch(ready_operations, time)

    n = len(ready_operations)

    # Precompute per-operation features
    proc_times = np.empty(n)
    remaining_works = np.empty(n)
    remaining_ops_arr = np.empty(n, dtype=int)
    job_ids = []

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        proc_times[i] = min(mo.values()) if mo else 0.0
        remaining_works[i] = op.get("remaining_work", 0.0)
        remaining_ops_arr[i] = op.get("remaining_ops", 0)
        job_ids.append(op.get("job", i))

    # Current global critical path: max remaining work across all ready jobs
    # Note: remaining_work already includes the current op's processing time
    # (it's the total remaining work for that job including this operation).
    # When we "complete" operation i, its job's remaining work decreases by proc_times[i].
    # But we only have info about ready operations. The global makespan lower bound
    # is at least the max remaining_work among all jobs. Since we only see ready ops,
    # we approximate: the critical path is max(remaining_works).

    current_cp = np.max(remaining_works) if n > 0 else 0.0

    # For each ready operation i, if we start it now, after it completes,
    # that job's remaining work becomes remaining_works[i] - proc_times[i].
    # The new critical path would be the max of:
    #   - (remaining_works[i] - proc_times[i]) for job i
    #   - remaining_works[j] for all other ready jobs j != i
    # (ignoring non-ready jobs which we don't have info on)

    # Compute reduction in critical path for each option
    reductions = np.zeros(n)

    for i in range(n):
        # New remaining work for job i after completing op i
        new_rw_i = remaining_works[i] - proc_times[i]
        # Max remaining work among all other ready ops
        if n > 1:
            others = np.delete(remaining_works, i)
            max_others = np.max(others)
        else:
            max_others = 0.0
        new_cp = max(new_rw_i, max_others)
        reductions[i] = current_cp - new_cp

    # We want the operation that gives the greatest reduction in critical path.
    # Tie-breaking: prefer shorter processing time, then fewer remaining ops.
    # Create a composite score: primary = reduction, secondary = -proc_time, tertiary = -remaining_ops
    # Use a small epsilon-based ranking to avoid floating point issues.

    # Sort by reduction descending, then proc_time ascending, then remaining_ops ascending
    # Use argsort with a stable approach
    order = np.lexsort((
        remaining_ops_arr,  # tertiary: fewer remaining ops first
        proc_times,          # secondary: shorter proc time first
        -reductions          # primary: greater reduction first
    ))

    best_idx = int(order[0])

    # Additional sanity check: if all reductions are ~0 (no improvement possible),
    # fall back to shortest processing time
    if np.all(np.abs(reductions) < 1e-12):
        best_idx = int(np.argmin(proc_times))

    return best_idx
# EVOLVE-BLOCK-END

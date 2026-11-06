"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 145.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - _time.time()


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Quick deadline guard: if we're close to timeout, return fastest-op tiebreak
    if _time_left() < 0.5:
        durations = []
        for op in ready_operations:
            mo = op["machine_options"]
            if mo:
                durations.append(min(mo.values()))
            else:
                durations.append(float('inf'))
        return int(np.argmin(durations))

    n = len(ready_operations)

    # Compute per-operation scores combining:
    # 1. Critical path: remaining_work of the job (longest tail)
    # 2. Processing time on best (shortest) machine option
    # 3. Number of remaining ops (urgency)
    # 4. Remaining work of the operation itself

    remaining_works = []
    min_durations = []
    remaining_ops_counts = []
    op_remaining_works = []

    for op in ready_operations:
        rw = op["remaining_work"]
        remaining_works.append(rw)
        mo = op["machine_options"]
        if mo:
            min_dur = min(mo.values())
        else:
            min_dur = float('inf')
        min_durations.append(min_dur)
        remaining_ops_counts.append(op["remaining_ops"])
        # The work for THIS operation is the sum of its machine options' durations? 
        # Actually, remaining_work is job-level. For op-level, use min duration as proxy.
        op_remaining_works.append(min_dur)

    remaining_works = np.array(remaining_works, dtype=np.float64)
    min_durations = np.array(min_durations, dtype=np.float64)
    remaining_ops_counts = np.array(remaining_ops_counts, dtype=np.float64)
    op_remaining_works = np.array(op_remaining_works, dtype=np.float64)

    # Normalize each component to [0, 1] for weighting
    def _normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    norm_rw = _normalize(remaining_works)
    norm_md = _normalize(min_durations)
    norm_ro = _normalize(remaining_ops_counts)
    norm_ow = _normalize(op_remaining_works)

    # Weighted score: prioritize high remaining_work (critical path),
    # low min_duration (quick completion), high remaining_ops (many ops left = urgent)
    # The idea: clear operations on jobs with most remaining work first,
    # preferring shorter processing times to keep the critical path moving.
    score = (
        0.50 * norm_rw          # most remaining job work first
        + 0.20 * (1.0 - norm_md)  # shorter processing time first
        + 0.20 * norm_ro         # more remaining ops = more urgent
        + 0.10 * (1.0 - norm_ow)  # shorter op work first
    )

    # Break ties by lower remaining_work of the operation's min duration
    # (already captured above), then by index for stability
    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END

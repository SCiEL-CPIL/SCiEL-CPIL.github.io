"""Combined Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Combine multiple heuristics into a composite score:
    1. Critical Path Length (CPL): Jobs with longer remaining paths are urgent.
    2. Minimum Remaining Work: Favors operations that finish quickly (shortest processing time).
    3. Machine Load Balance: Prefer less busy machines if we can infer from options.
    4. Number of remaining ops: More remaining ops = higher priority (tail of the job).
    
    The composite score is:
      score = w1 * normalized_remaining_work + 
              w2 * (1 / (min_duration + epsilon)) + 
              w3 * (1 / (remaining_ops + 1))
    
    We pick the operation with the highest score.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features using numpy for efficiency
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each operation, compute true min duration and machine flexibility
    min_durations = np.zeros(n, dtype=np.float64)
    second_min_durations = np.zeros(n, dtype=np.float64)
    n_machines = np.zeros(n, dtype=np.float64)
    duration_spread = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durs = np.array(list(mo.values()), dtype=np.float64)
            n_machines[i] = len(durs)
            min_durations[i] = durs.min()
            if len(durs) > 1:
                sorted_durs = np.sort(durs)
                second_min_durations[i] = sorted_durs[1]
                duration_spread[i] = durs.max() - durs.min()
            else:
                second_min_durations[i] = durs[0]
                duration_spread[i] = 0.0
        else:
            min_durations[i] = 1.0
            second_min_durations[i] = 1.0
            n_machines[i] = 1.0
            duration_spread[i] = 0.0

    # Normalize remaining_work to [0, 1]
    max_rw = remaining_works.max()
    min_rw = remaining_works.min()
    rw_range = max_rw - min_rw
    if rw_range < 1e-9:
        norm_rw = np.full(n, 0.5)
    else:
        norm_rw = (remaining_works - min_rw) / rw_range

    # Inverse min duration: shorter processing time gets higher score
    # Use log transform to dampen extreme values
    inv_dur = 1.0 / (min_durations + 1e-6)
    max_inv = inv_dur.max()
    if max_inv < 1e-9:
        norm_inv_dur = np.full(n, 0.5)
    else:
        norm_inv_dur = inv_dur / max_inv
    # Apply log compression to reduce sensitivity to extreme short durations
    norm_inv_dur = np.log1p(norm_inv_dur) / np.log1p(1.0)  # normalize to [0,1]
    
    # Flexibility bonus: gap between min and second-min indicates a clearly preferred machine
    # Larger gap = more constrained to one machine = slightly higher urgency
    min_gaps = second_min_durations - min_durations
    max_gap = min_gaps.max()
    if max_gap < 1e-9:
        norm_gap = np.full(n, 0.5)
    else:
        norm_gap = min_gaps / max_gap

    # Normalize remaining_ops: more remaining ops = higher urgency (critical path tail)
    max_ro = remaining_ops.max()
    min_ro = remaining_ops.min()
    ro_range = max_ro - min_ro
    if ro_range < 1e-9:
        norm_ro = np.full(n, 0.5)
    else:
        norm_ro = (remaining_ops - min_ro) / ro_range

    # Machine flexibility: fewer machine options = more constrained = higher priority
    # This acts as a tie-breaker for constrained operations
    max_nm = n_machines.max()
    min_nm = n_machines.min()
    nm_range = max_nm - min_nm
    if nm_range < 1e-9:
        norm_constraint = np.full(n, 0.5)
    else:
        # Invert: fewer machines = higher constraint score
        norm_constraint = 1.0 - (n_machines - min_nm) / nm_range

    # Duration spread: operations with similar durations across machines are more flexible
    # Lower spread = more flexibility = slightly lower urgency
    max_spread = duration_spread.max()
    if max_spread < 1e-9:
        norm_spread = np.full(n, 0.5)
    else:
        norm_spread = duration_spread / max_spread

    # Composite score weights - tuned for job-shop critical path dominance
    w1 = 0.50  # remaining work (critical path) - dominant factor
    w2 = 0.25  # shortest processing time
    w3 = 0.12  # remaining ops urgency
    w4 = 0.08  # machine constraint (fewer options = higher priority)
    w5 = 0.05  # machine preference gap (larger gap = more constrained)

    scores = w1 * norm_rw + w2 * norm_inv_dur + w3 * norm_ro + w4 * norm_constraint + w5 * norm_gap

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END

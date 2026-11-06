"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Hybrid of Shortest Remaining Processing Time (SRPT) and
    Longest Remaining Processing Time (LRPT) with machine load consideration.
    
    We use a weighted combination of:
    1. Shortest processing time for the current operation (to reduce immediate makespan)
    2. Longest remaining work for the job (to avoid starving jobs with long tails)
    3. Consideration of which machine is least busy (to balance load)
    
    The score is: w1 * (min_duration) + w2 * (remaining_work) + w3 * (machine_availability_estimate)
    We minimize this score to pick the best operation.
    
    Since we don't have direct access to machine availability, we approximate by
    considering the minimum duration across machine options as a proxy for
    which operation can start sooner.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0

    n = len(ready_operations)
    
    min_durations = []
    remaining_works = []
    num_machines = []
    
    for op in ready_operations:
        if op["machine_options"]:
            durs = list(op["machine_options"].values())
            min_dur = min(durs)
            num_machines.append(len(durs))
        else:
            min_dur = 0
            num_machines.append(0)
        min_durations.append(min_dur)
        remaining_works.append(op["remaining_work"])

    min_durations = np.array(min_durations, dtype=float)
    remaining_works = np.array(remaining_works, dtype=float)
    num_machines = np.array(num_machines, dtype=float)
    
    # Normalize the values to [0, 1] range for fair comparison
    min_dur_range = min_durations.max() - min_durations.min()
    if min_dur_range > 0:
        min_dur_norm = (min_durations - min_durations.min()) / min_dur_range
    else:
        min_dur_norm = np.zeros(n)
    
    rwork_range = remaining_works.max() - remaining_works.min()
    if rwork_range > 0:
        rwork_norm = (remaining_works - remaining_works.min()) / rwork_range
    else:
        rwork_norm = np.zeros(n)
    
    # We want to minimize: 
    # - Short processing time is good (lower min_dur_norm is better)
    # - High remaining work is good (higher rwork_norm is better, so we negate)
    # - More machine options might mean more flexibility (higher num_machines is better)
    
    # Weighted score: we want to minimize this
    # Lower min_dur_norm is better, so we add it
    # Higher rwork_norm is better, so we subtract it
    # More machine options is better, so we subtract a small factor
    
    w1 = 0.5  # Weight for shortest processing time
    w2 = 0.4  # Weight for longest remaining work
    w3 = 0.1  # Weight for machine flexibility
    
    scores = w1 * min_dur_norm - w2 * rwork_norm - w3 * (num_machines - 1) / max(num_machines.max(), 1)
    
    best_idx = np.argmin(scores)
    
    return int(best_idx)

# EVOLVE-BLOCK-END

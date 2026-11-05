"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def _soft_min(values, beta=10.0):
    """
    Smooth approximation of min using log-sum-exp.
    For small beta, approaches true min. For large beta, smoother.
    We use a negative log-sum-exp to get soft-min.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    arr = np.array(values, dtype=np.float64)
    # soft-min = -soft-max(-x)
    # soft-max(x) = (1/beta) * log(sum(exp(beta * x)))
    neg_arr = -arr
    max_neg = np.max(neg_arr)
    exp_sum = np.sum(np.exp(beta * (neg_arr - max_neg)))
    soft_max_neg = (1.0 / beta) * np.log(exp_sum) + max_neg
    return -soft_max_neg


def _soft_max(values, beta=10.0):
    """
    Smooth approximation of max using log-sum-exp.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    arr = np.array(values, dtype=np.float64)
    max_val = np.max(arr)
    exp_sum = np.sum(np.exp(beta * (arr - max_val)))
    return (1.0 / beta) * np.log(exp_sum) + max_val


def _best_machine_for_op(op, rw):
    """
    For a given operation, select the machine that minimizes duration.
    With a slight preference for machines that have less "load" implied by
    the duration itself (shorter duration = less machine time consumed).
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Primary: Job with most remaining work (LRPT) to prioritize critical path.
    2. Secondary: Smooth surrogate combining min/max duration and remaining ops.
    3. Machine: Machine with shortest duration for the chosen operation (SPT).
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op, op["remaining_work"])
        return (0, best_m)

    # Deadline guard: if we are too close to the limit, return a quick heuristic
    if _elapsed() > 145:
        # Quick fallback: pick first op, min duration machine
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op, op["remaining_work"])
        return (0, best_m)

    scores = []
    best_machines = []
    
    # Precompute normalization factors
    all_rw = [op["remaining_work"] for op in ready_operations]
    max_rw = max(all_rw) if all_rw else 1.0
    if max_rw <= 0:
        max_rw = 1.0

    for op in ready_operations:
        rw = op["remaining_work"]
        durations = op["machine_options"]
        if not durations:
            scores.append(-1)
            best_machines.append(None)
            continue
            
        dur_values = list(durations.values())
        min_dur = min(dur_values)
        max_dur = max(dur_values)
        
        # Smooth surrogate: soft-min and soft-max provide differentiability
        # and better discrimination than hard min/max
        sm_min = _soft_min(dur_values, beta=5.0)
        sm_max = _soft_max(dur_values, beta=5.0)
        
        # Remaining ops: fewer remaining ops = closer to completion = slight urgency
        remaining_ops = op.get("remaining_ops", 0)
        
        # Composite score:
        # - Primary: remaining work (normalized, dominant)
        # - Secondary: prefer operations where the "effective" duration is moderate
        #   (not too short which might indicate low criticality, not too long which blocks)
        #   Use a concave function of the soft-min to reward moderate durations
        # - Tertiary: urgency from remaining_ops (fewer = higher priority slightly)
        
        # Normalize rw to [0, 1] range for cleaner weighting
        rw_norm = rw / max_rw
        
        # Duration component: we want to prioritize ops that are "critical" 
        # A long min_dur on a high-rw job is very critical (long blocking op)
        # A short min_dur on a high-rw job is also good (quick win)
        # Use: rw_norm * (sm_min + 0.1 * sm_max) to weight by criticality
        
        # Urgency: 1/(remaining_ops + 1) gives higher score for fewer remaining ops
        urgency = 1.0 / (remaining_ops + 1.0)
        
        # Final score:
        # Dominant: rw_norm
        # Secondary: duration criticality (soft-min scaled by rw_norm)
        # Tertiary: urgency
        score = rw_norm * 1e6 + rw_norm * sm_min + 0.5 * urgency * (1.0 / (sm_min + 1e-9))
        
        scores.append(score)
        
        best_m, _ = _best_machine_for_op(op, rw)
        best_machines.append(best_m)

    # Find index with max score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    if best_machine is None:
        return best_idx
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END

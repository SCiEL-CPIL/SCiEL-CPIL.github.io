"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from math import log, exp

# Deadline guard: evaluator kills after 150s; keep a safety margin.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin


def _time_remaining():
    return _DEADLINE - _time.time()


def _soft_min(values, temperature=0.1):
    """
    Soft minimum using log-sum-exp trick.
    For small temperature, approximates min(values).
    Returns a smooth surrogate that accounts for all machine options,
    not just the single best one.
    """
    if len(values) == 1:
        return values[0]
    
    # Shift for numerical stability
    min_val = min(values)
    shifted = [v - min_val for v in values]
    
    # Compute log-sum-exp of exp(-v/temperature)
    # This gives a soft minimum
    neg_shifted = [-v / temperature for v in shifted]
    max_neg = max(neg_shifted)
    
    sum_exp = sum(exp(n - max_neg) for n in neg_shifted)
    log_sum_exp = max_neg + log(sum_exp) if sum_exp > 0 else 0.0
    
    # Soft min = -temperature * log(sum(exp(-v/temperature))) + min_val
    # But we want it to be close to min when temperature is small
    soft_min = min_val - temperature * log(sum_exp) if sum_exp > 0 else min_val
    
    return max(soft_min, 0.001)


def dispatch(ready_operations, time):
    """
    Dynamic bottleneck-aware dispatch rule with smooth surrogates.
    
    Strategy: For each ready operation, compute a soft-min duration across
    all machine options (not just the hard min). Score each operation by:
      1. WSPT urgency (remaining_work / soft_min_duration)
      2. Machine load balancing using soft-min estimates
      3. Critical-path bonus for jobs with more remaining work
    
    We select the operation that maximizes a composite score designed to
    minimize makespan by balancing urgency with machine utilization.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _time_remaining() < 0.5:
        # Fallback: simple WSPT with hard min
        scores = []
        for op in ready_operations:
            min_dur = min(op["machine_options"].values())
            scores.append(op["remaining_work"] / max(min_dur, 0.001))
        return int(np.argmax(scores))

    # Extract features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Compute soft-min durations for each operation
    soft_min_durs = []
    hard_min_durs = []
    best_machines = []
    
    for op in ready_operations:
        mo = op["machine_options"]
        durs = list(mo.values())
        hard_min = min(durs)
        hard_min_durs.append(hard_min)
        
        # Use soft-min with small temperature for smoothness
        soft_min = _soft_min(durs, temperature=0.05)
        soft_min_durs.append(soft_min)
        
        # Track best machine (hard min) for load estimation
        best_m = min(mo.keys(), key=lambda m: mo[m])
        best_machines.append(best_m)
    
    soft_min_durs = np.array(soft_min_durs)
    hard_min_durs = np.array(hard_min_durs)
    
    # Avoid division by zero
    soft_min_safe = np.where(soft_min_durs > 0.001, soft_min_durs, 1.0)
    hard_min_safe = np.where(hard_min_durs > 0.001, hard_min_durs, 1.0)
    
    # WSPT ratio using soft-min (smoother, accounts for alternatives)
    wspt_soft = remaining_works / soft_min_safe
    
    # Also compute WSPT with hard min for comparison
    wspt_hard = remaining_works / hard_min_safe
    
    # Compute machine load using soft-min durations
    # This gives a more accurate picture of machine congestion
    machine_load = {}
    for i in range(n):
        best_m = best_machines[i]
        # Use soft-min duration as estimate of time this op will occupy machine
        load = soft_min_durs[i]
        machine_load[best_m] = machine_load.get(best_m, 0.0) + load
    
    # Normalize machine loads
    total_load = sum(machine_load.values()) if machine_load else 1.0
    machine_load_norm = {}
    for m, load in machine_load.items():
        machine_load_norm[m] = load / total_load if total_load > 0 else 0.0
    
    # Critical path bonus: jobs with more remaining ops are more urgent
    # Normalize remaining_ops to [0, 1]
    max_rem_ops = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    crit_bonus = remaining_ops / max_rem_ops
    
    # Score: combine WSPT (soft) with machine load balancing and critical path
    # Weight: 0.6 * WSPT_soft + 0.25 * (1 - load) * WSPT_hard + 0.15 * crit_bonus
    scores = np.zeros(n)
    for i in range(n):
        best_m = best_machines[i]
        m_load = machine_load_norm.get(best_m, 0.0)
        
        # Base urgency from soft-min WSPT
        base_urgency = wspt_soft[i]
        
        # Load-balancing adjustment: prefer less-loaded machines
        load_factor = (1.0 - 0.4 * m_load)
        
        # Critical path bonus
        cp_bonus = 0.15 * crit_bonus[i]
        
        # Combine
        scores[i] = base_urgency * load_factor + cp_bonus
    
    # Final selection
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END

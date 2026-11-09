"""Job-Shop Scheduling Dispatch Rule.
Uses a heuristic combining Criticality (Remaining Work), 
Efficiency (Shortest Processing Time), and Urgency (Fewest Remaining Ops).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0  # Safety margin against 120s/150s limit

def _soft_min(durations, temperature=0.5):
    """Smooth surrogate for min using log-sum-exp trick.
    Returns a value close to min but differentiable/smooth."""
    if len(durations) == 0:
        return 1.0
    d = np.array(durations, dtype=np.float64)
    d_min = np.min(d)
    # Soft min: -T * log(sum(exp(-d/T)))
    # As T -> 0, approaches hard min
    scaled = -d / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        # If only one option, pick it and its best machine
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Deadline check: if we are running out of time, use a simple fallback
    if _time.time() - _START_TS > DEADLINE_S:
        # Fallback: Pick the one with the shortest min duration
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(float('inf'))
        idx = int(np.argmin(min_durs))
        op = ready_operations[idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (idx, best_machine)
        return idx

    # Extract features for vectorized computation
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op.get("remaining_ops", 1) for op in ready_operations], dtype=np.float64)
    
    # Calculate soft-min duration and best machine for each operation
    soft_min_durations = np.zeros(n, dtype=np.float64)
    hard_min_durations = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            durs = list(machine_options.values())
            # Soft min for smooth scoring
            soft_min_durations[i] = _soft_min(durs, temperature=0.3)
            # Hard min for actual machine selection
            best_m, min_d = min(machine_options.items(), key=lambda item: item[1])
            hard_min_durations[i] = min_d
            best_machines[i] = best_m
        else:
            soft_min_durations[i] = 1.0
            hard_min_durations[i] = 1.0
            best_machines[i] = None

    # Avoid division by zero
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)
    soft_min_durations_safe = soft_min_durations + 1e-9

    # Heuristic Score:
    # 1. Criticality: Higher remaining_work is better.
    # 2. Efficiency: Lower soft_min_duration is better (smooth surrogate).
    # 3. Urgency: Lower remaining_ops is better (jobs closer to completion).
    # 4. Critical path proxy: remaining_work / remaining_ops (avg work per op)
    
    # Normalization to make weights comparable
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    smd_min = np.min(soft_min_durations_safe)
    smd_max = np.max(soft_min_durations_safe)
    
    # Normalize soft-min durations to [0, 1] where 1 is best (shortest)
    if smd_max > smd_min:
        norm_dur = (smd_max - soft_min_durations_safe) / (smd_max - smd_min)
    else:
        norm_dur = np.ones(n)
        
    # Normalize remaining ops to [0, 1] where 1 is best (fewest)
    max_ops = np.max(remaining_ops_safe) if np.max(remaining_ops_safe) > 0 else 1.0
    norm_ops = (max_ops - remaining_ops_safe) / max_ops
    
    # Normalize remaining work to [0, 1] where 1 is best (most)
    norm_work = remaining_works / max_work

    # Critical path proxy: higher avg work per remaining op = more critical
    avg_work_per_op = remaining_works / remaining_ops_safe
    max_avg = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    norm_critical = avg_work_per_op / max_avg

    # Weighted Sum
    # Weights: Criticality (0.45), Efficiency (0.25), Urgency (0.15), Critical path (0.15)
    w_work = 0.45
    w_dur = 0.25
    w_ops = 0.15
    w_critical = 0.15
    
    scores = (w_work * norm_work) + (w_dur * norm_dur) + (w_ops * norm_ops) + (w_critical * norm_critical)
    
    best_idx = int(np.argmax(scores))
    
    # Return index and the best machine for that operation
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx

# EVOLVE-BLOCK-END

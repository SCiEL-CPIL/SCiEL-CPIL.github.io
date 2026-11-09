import time as _time
import math

_START_TIME = None
_DEADLINE = 150.0

def _soft_min(values, tau=1.0):
    """Compute soft-min using log-sum-exp for smooth surrogate of min."""
    if not values:
        return 1.0
    v_min = min(values)
    if v_min <= 0:
        return v_min
    # Soft min: -tau * log(sum(exp(-v/tau)))
    # Use numerical stability
    scaled = [-v / tau for v in values]
    max_scaled = max(scaled)
    sum_exp = sum(math.exp(s - max_scaled) for s in scaled)
    if sum_exp == 0:
        return v_min
    return -tau * (max_scaled + math.log(sum_exp))

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    if _time.time() - _START_TIME > _DEADLINE - 5.0:
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Pre-calculate features
    max_rw = 0.0
    min_min_d = float('inf')
    max_min_d = 0.0
    min_n_machines = float('inf')
    sum_rw = 0.0
    min_remaining_ops = float('inf')
    
    features = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            # Soft-min captures distribution better than hard min
            soft_min_d = _soft_min(durations, tau=0.5)
            n_machines = len(mo)
            best_machine = min(mo, key=mo.get)
        else:
            min_d = 1.0
            soft_min_d = 1.0
            n_machines = 1
            best_machine = None

        features.append({
            'rw': rw,
            'min_d': min_d,
            'soft_min_d': soft_min_d,
            'n_machines': n_machines,
            'best_machine': best_machine,
            'remaining_ops': remaining_ops
        })
        
        if rw > max_rw:
            max_rw = rw
        sum_rw += rw
        if min_d < min_min_d:
            min_min_d = min_d
        if min_d > max_min_d:
            max_min_d = min_d
        if n_machines < min_n_machines:
            min_n_machines = n_machines
        if remaining_ops < min_remaining_ops:
            min_remaining_ops = remaining_ops

    # Handle edge cases for normalization
    if max_rw == 0: max_rw = 1.0
    if min_min_d == 0: min_min_d = 1.0
    if max_min_d == 0: max_min_d = 1.0
    if min_n_machines == 0: min_n_machines = 1.0
    if min_remaining_ops == float('inf'): min_remaining_ops = 1
    avg_rw = sum_rw / n if n > 0 else 1.0

    best_score = -1.0
    best_idx = 0
    best_machine = None

    for i in range(n):
        f = features[i]
        
        # Component 1: Remaining Work (Higher is better)
        # Use relative to average for better discrimination
        score_rw = f['rw'] / max_rw
        
        # Component 2: Efficiency using soft-min (captures distribution)
        # Lower soft_min_d is better
        score_eff = min_min_d / f['soft_min_d']
        
        # Component 3: Flexibility - reward more machine options to reduce bottlenecks
        # Higher n_machines is better (more flexibility)
        max_n_machines = max(feat['n_machines'] for feat in features)
        if max_n_machines == 0: max_n_machines = 1
        score_flex = f['n_machines'] / max_n_machines
        
        # Component 4: Urgency - fewer remaining ops means this job is closer to completion
        # Higher urgency is better (complete jobs sooner to free resources)
        max_remaining_ops = max(feat['remaining_ops'] for feat in features)
        if max_remaining_ops == 0: max_remaining_ops = 1
        score_urgency = 1.0 - (f['remaining_ops'] / max_remaining_ops)
        
        # Weights tuned based on measured improvements
        w_rw = 0.35
        w_eff = 0.25
        w_flex = 0.25
        w_urgency = 0.15
        
        total_score = w_rw * score_rw + w_eff * score_eff + w_flex * score_flex + w_urgency * score_urgency
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            best_machine = f['best_machine']
        elif abs(total_score - best_score) < 1e-9:
            # Tie-breaker: Prefer shorter duration
            if f['min_d'] < features[best_idx]['min_d']:
                best_idx = i
                best_machine = f['best_machine']

    # Return (index, machine) to ensure we pick the optimal machine for the chosen op
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx

# EVOLVE-BLOCK-END

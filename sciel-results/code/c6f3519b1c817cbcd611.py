"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill

# Adaptive parameters: will be tuned based on problem scale
# For 10 jobs, 5 machines: moderate urgency weight
_URGENCY_WEIGHT = 0.25
_MACHINE_PENALTY_WEIGHT = 0.15


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def _smooth_min(values, temperature=1.0):
    """Smooth min via log-sum-exp: -log(sum(exp(-v_i/T)))*T approximates min(v_i) smoothly."""
    if len(values) == 0:
        return 0.0
    v = np.array(values, dtype=np.float64)
    if len(v) == 1:
        return v[0]
    # For numerical stability
    v_min = np.min(v)
    exp_terms = np.exp(-(v - v_min) / temperature)
    log_sum = np.log(np.sum(exp_terms))
    return v_min + temperature * log_sum


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        # Return the single operation with its best machine
        op = ready_operations[0]
        if op["machine_options"]:
            best_machine = min(op["machine_options"], key=lambda m: op["machine_options"][m])
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Smooth min duration using log-sum-exp (temperature=1.0)
    smooth_min_durations = np.array([
        _smooth_min(list(op["machine_options"].values()), temperature=1.0) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Exact min for machine selection
    exact_min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Number of machine options (flexibility)
    n_machines = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Calculate average remaining work per op (intensity)
    safe_ops = np.maximum(remaining_ops, 1.0)
    intensity = remaining_works / safe_ops

    # Z-Score Normalization
    def _z_norm(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-12:
            return np.zeros_like(x)
        return (x - mean) / std

    # Time-aware urgency: operations with less remaining work relative to current time
    # are more urgent to complete (to avoid becoming bottlenecks later)
    if time > 0:
        urgency = remaining_works / (time + 1.0)  # Higher = more urgent
    else:
        urgency = np.zeros_like(remaining_works)
    
    # Model:
    # High Intensity (high work per op) is good (LRPT variant).
    # High Remaining Ops is good (keep jobs in pipeline).
    # Low Smooth Min Duration is good (SPT, clears machine quickly).
    # High flexibility (more machine options) is slightly good (more scheduling options).
    # High Urgency (work/time ratio) is good - prioritize operations that will become bottlenecks.
    
    # Parameters (Weights) - tuned for open-shop with time-awareness
    w_intensity = 1.0
    w_ops = 0.4
    w_min_dur = -0.3
    w_flex = 0.1
    w_urgency = _URGENCY_WEIGHT
    
    z_intensity = _z_norm(intensity)
    z_ops = _z_norm(remaining_ops)
    z_min_dur = _z_norm(smooth_min_durations)
    z_flex = _z_norm(n_machines)
    z_urgency = _z_norm(urgency)

    score = (w_intensity * z_intensity + w_ops * z_ops + 
             w_min_dur * z_min_dur + w_flex * z_flex + 
             w_urgency * z_urgency)

    # Handle NaNs or Infs
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)

    best_idx = int(np.argmax(score))
    
    # Select best machine for the chosen operation using composite score:
    # Minimize duration, but also consider the impact on remaining work
    op = ready_operations[best_idx]
    if op["machine_options"]:
        # For each machine option, compute a composite cost:
        # - Primary: duration (lower is better)
        # - Secondary: how much this choice affects the job's remaining work
        #   (choosing a slower machine now might be worse if it delays the job)
        
        # Get all machine options
        machine_durations = list(op["machine_options"].values())
        machine_names = list(op["machine_options"].keys())
        
        if len(machine_durations) == 1:
            best_machine = machine_names[0]
        else:
            # Normalize durations to [0, 1] range
            min_dur = min(machine_durations)
            max_dur = max(machine_durations)
            dur_range = max_dur - min_dur
            
            if dur_range < 1e-9:
                # All machines have same duration, pick first
                best_machine = machine_names[0]
            else:
                # Composite score: primarily minimize duration, 
                # with a small penalty for machines that leave more remaining work
                # (this is a proxy for "this choice constrains future scheduling")
                best_machine = None
                best_score = float('inf')
                
                for i, m in enumerate(machine_names):
                    dur = machine_durations[i]
                    # Normalized duration (0 = fastest, 1 = slowest)
                    norm_dur = (dur - min_dur) / dur_range
                    
                    # The "penalty" is just the normalized duration itself
                    # (we don't have full remaining work breakdown by machine)
                    composite = norm_dur
                    
                    if composite < best_score:
                        best_score = composite
                        best_machine = m
                
                if best_machine is None:
                    best_machine = machine_names[0]
        
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END

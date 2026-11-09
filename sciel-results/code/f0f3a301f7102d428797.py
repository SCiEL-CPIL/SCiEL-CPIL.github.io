import numpy as np
import time as _time

# Deadline guard to ensure we finish well before the 150s limit
_DEADLINE = 145.0
_START_TIME = _time.time()

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Check deadline
    if _time.time() - _START_TIME > _DEADLINE:
        return 0

    # Extract metrics
    works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    ops_counts = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    min_durs = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            min_durs[i] = 0.0
        else:
            min_durs[i] = min(opts.values())
    
    # Handle edge case where all works are 0
    if works.max() == 0:
        return 0

    # Log transform to handle skewness in processing times
    log_works = np.log(works + 1.0)
    log_durs = np.log(min_durs + 1.0)
    
    # Z-Score Normalization for robustness against outliers
    def z_score(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std
        
    z_work = z_score(log_works)
    z_dur = z_score(log_durs)
    z_ops = z_score(ops_counts)
    
    # Calculate Coefficient of Variation for works to determine adaptive weighting
    mean_work = works.mean()
    if mean_work > 0:
        std_work = works.std()
        cv = std_work / mean_work
    else:
        cv = 0.0
        
    # Clamp CV to [0, 1] for stable weighting
    cv_clamped = np.clip(cv, 0.0, 1.0)
    
    # Adaptive weights:
    # High CV (large disparity) -> Prioritize MWRF (Work)
    # Low CV (similar work) -> Prioritize SPT (Duration)
    w_work = 0.5 + 0.5 * cv_clamped
    w_spt = 1.0 - w_work
    
    # Small weight for remaining ops to break ties and favor complex jobs
    w_ops = 0.05
    
    # Total score: Higher is better
    # We maximize work, minimize duration (so subtract z_dur), maximize ops
    scores = w_work * z_work - w_spt * z_dur + w_ops * z_ops
    
    best_idx = int(np.argmax(scores))
    return best_idx

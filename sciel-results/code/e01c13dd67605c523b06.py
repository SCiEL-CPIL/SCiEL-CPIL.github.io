"""Rewritten dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s limit


def _is_deadline_exceeded():
    return _time.time() > _DEADLINE


def _soft_min(values, temperature=1.0):
    """
    Soft minimum using log-sum-exp trick for numerical stability.
    Returns a smooth approximation of min(values).
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # Subtract max for numerical stability in log-sum-exp
    max_val = max(values)
    if max_val == 0:
        return 0.0
    
    # For soft min: -log-sum-exp(-x/temperature) * temperature
    scaled = [-v / temperature for v in values]
    max_scaled = max(scaled)
    sum_exp = sum(math.exp(s - max_scaled) for s in scaled)
    log_sum_exp = max_scaled + math.log(sum_exp)
    
    return -log_sum_exp * temperature


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Uses a balanced multi-criteria heuristic tuned for job-shop scheduling:
    1. Prioritize jobs with more remaining work (reduces makespan by keeping pipeline full)
    2. Among similar work amounts, prefer shorter processing times (reduces idle time)
    3. Break ties by fewer remaining operations (completes jobs sooner)
    
    Returns the index of the chosen operation.
    """
    if not ready_operations:
        return 0

    if _is_deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Collect features for all operations
    features = []
    for i, op in enumerate(ready_operations):
        options = op.get('machine_options', {})
        if not options:
            features.append((0, 0, 0, []))  # Low priority
            continue

        durations = list(options.values())
        min_dur = min(durations)
        # Use soft-min for smoother duration preference
        soft_min_dur = _soft_min(durations, temperature=2.0)
        remaining_work = op.get('remaining_work', 0)
        remaining_ops = op.get('remaining_ops', 0)
        features.append((remaining_work, min_dur, remaining_ops, soft_min_dur))

    # Find max remaining work for normalization
    max_work = max(f[0] for f in features) if features else 1
    min_min_dur = min(f[1] for f in features) if features else 1
    max_min_dur = max(f[1] for f in features) if features else 1
    max_remaining_ops = max(f[2] for f in features) if features else 1
    max_soft_dur = max(f[3] for f in features) if features else 1
    min_soft_dur = min(f[3] for f in features) if features else 1

    best_idx = 0
    best_score = float('-inf')

    for i, (rw, md, ro, smd) in enumerate(features):
        if rw == 0 and md == 0 and ro == 0:
            continue

        # Normalize to [0, 1]
        norm_work = rw / (max_work + 1e-9) if max_work > 0 else 0
        # For duration: shorter is better, so invert
        norm_dur = 1.0 - (md - min_min_dur) / (max_min_dur - min_min_dur + 1e-9) if max_min_dur > min_min_dur else 1.0
        # For remaining ops: fewer is better
        norm_ops = 1.0 - ro / (max_remaining_ops + 1e-9) if max_remaining_ops > 0 else 1.0
        # Soft-min duration: lower is better (shorter effective processing time)
        norm_smd = 1.0 - (smd - min_soft_dur) / (max_soft_dur - min_soft_dur + 1e-9) if max_soft_dur > min_soft_dur else 1.0

        # Weighted score: emphasize remaining work most, then duration, then ops
        # Weights tuned for job-shop: work matters most for makespan
        # Add soft-min duration as an additional signal for smoother optimization
        score = 0.5 * norm_work + 0.25 * norm_dur + 0.1 * norm_ops + 0.15 * norm_smd

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END

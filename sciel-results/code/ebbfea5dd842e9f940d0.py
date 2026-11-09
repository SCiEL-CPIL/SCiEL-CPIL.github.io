"""Rewritten program for Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit


def _soft_min(values, temperature=0.5):
    """Compute soft minimum using log-sum-exp trick."""
    if not values:
        return 0.0
    vmin = min(values)
    if vmin <= 1e-9:
        return vmin
    # Soft min: -T * log(sum(exp(-v/T)))
    scaled = [-v / temperature for v in values]
    max_scaled = max(scaled)
    log_sum = math.log(sum(math.exp(s - max_scaled) for s in scaled)) + max_scaled
    return -temperature * log_sum


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Use a ratio-based heuristic that balances:
    1. Critical Path Length (CPL) via remaining_work: higher is more urgent.
    2. Shortest Processing Time (SPT): shorter min duration preferred.
    3. Machine contention: prefer ops on less-contended machines.
    4. Remaining operations count as a tiebreaker.
    
    We compute a ratio score (urgency / effort) and pick the operation with the highest score.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    
    # For each operation, compute soft-min duration (accounts for machine flexibility)
    # Also track which machines are contested
    soft_min_durations = []
    machine_counts = {}
    
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            machines = list(mo.keys())
            # Count machine contention
            for m in machines:
                machine_counts[m] = machine_counts.get(m, 0) + 1
            # Soft min for flexibility-aware duration
            smd = _soft_min(durations, temperature=0.3)
            soft_min_durations.append(smd)
        else:
            soft_min_durations.append(1.0)
    
    # Compute contention score: for each op, average contention of its available machines
    # Lower contention = better (less likely to wait)
    contention_scores = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            machines = list(mo.keys())
            avg_contention = sum(machine_counts[m] for m in machines) / len(machines)
        else:
            avg_contention = 1.0
        contention_scores.append(avg_contention)

    # Ratio-based scoring: remaining_work / (1 + soft_min_duration)
    # This captures urgency-to-efficiency tradeoff
    # Higher remaining work = more urgent, shorter duration = easier to complete
    ratio_scores = []
    for i in range(n):
        rw = remaining_works[i]
        smd = soft_min_durations[i]
        # Add small epsilon to avoid division by zero
        ratio = rw / (1.0 + smd)
        ratio_scores.append(ratio)
    
    # Normalize ratio scores to [0, 1]
    max_ratio = max(ratio_scores) if ratio_scores else 1.0
    min_ratio = min(ratio_scores) if ratio_scores else 0.0
    ratio_range = max_ratio - min_ratio
    if ratio_range < 1e-9:
        norm_ratio = [0.5] * n
    else:
        norm_ratio = [(r - min_ratio) / ratio_range for r in ratio_scores]
    
    # Inverse contention: lower contention is better
    max_cont = max(contention_scores) if contention_scores else 1.0
    min_cont = min(contention_scores) if contention_scores else 1.0
    cont_range = max_cont - min_cont
    if cont_range < 1e-9:
        norm_cont = [0.5] * n
    else:
        norm_cont = [1.0 - (c - min_cont) / cont_range for c in contention_scores]

    # Remaining ops: more remaining ops means the job has more future work to unlock
    # So we prioritize jobs with more remaining ops to keep the schedule flowing
    max_ro = max(remaining_ops) if remaining_ops else 1.0
    min_ro = min(remaining_ops) if remaining_ops else 0.0
    ro_range = max_ro - min_ro
    if ro_range < 1e-9:
        norm_ro = [0.5] * n
    else:
        norm_ro = [(ro - min_ro) / ro_range for ro in remaining_ops]

    # Composite score weights - tuned for job-shop scheduling
    # Higher weight on ratio (urgency/efficiency) as it directly impacts makespan
    # Machine contention: prefer ops on less-contended machines
    # Lower weight on remaining ops as a tiebreaker
    w1 = 0.70  # ratio score (urgency/efficiency) - dominant factor
    w2 = 0.25  # machine contention (inverse)
    w3 = 0.05  # remaining ops urgency

    scores = []
    for i in range(n):
        s = w1 * norm_ratio[i] + w2 * norm_cont[i] + w3 * norm_ro[i]
        scores.append(s)

    best_idx = max(range(n), key=lambda i: scores[i])
    return int(best_idx)
# EVOLVE-BLOCK-END

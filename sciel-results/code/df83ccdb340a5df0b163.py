"""Rewritten program for Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Use a composite heuristic that balances:
    1. Critical Path Length (CPL) via remaining_work: higher is more urgent.
    2. Shortest Processing Time (SPT): shorter min duration preferred.
    3. Remaining operations count: more remaining ops means the job is in early stages,
       so completing the current op unlocks more future work - prioritize jobs with
       more remaining ops to keep the pipeline flowing.
    4. Machine preference: prefer machines with shorter durations to reduce makespan.
    
    We compute a weighted score and pick the operation with the highest score.
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
    
    # For each operation, compute a soft-min over machine durations (log-sum-exp with
    # negative temperature) as a smooth surrogate for the hard min. This avoids the
    # discontinuity of min() when two machines have near-equal durations and gives a
    # slightly conservative (larger) effective duration when many options are available,
    # which can discourage over-optimistic SPT scoring.
    soft_min_durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durs = list(mo.values())
            # soft-min: -1/T * log(sum(exp(-d/T)))
            T = 0.5
            neg_d = [-d / T for d in durs]
            m = max(neg_d)
            s = sum(math.exp(x - m) for x in neg_d)
            soft_min = -T * (m + math.log(s))
        else:
            soft_min = 1.0  # fallback
        soft_min_durations.append(soft_min)

    # Normalize remaining_work to [0, 1] - higher remaining work = higher priority
    max_rw = max(remaining_works) if remaining_works else 1.0
    min_rw = min(remaining_works) if remaining_works else 0.0
    rw_range = max_rw - min_rw
    if rw_range < 1e-9:
        norm_rw = [0.5] * n
    else:
        norm_rw = [(rw - min_rw) / rw_range for rw in remaining_works]

    # Inverse soft-min duration: shorter processing time gets higher score
    # Scale by 1/soft_min_dur, but cap to avoid extreme values
    inv_dur = [1.0 / (md + 1e-6) for md in soft_min_durations]
    max_inv = max(inv_dur) if inv_dur else 1.0
    if max_inv < 1e-9:
        norm_inv_dur = [0.5] * n
    else:
        norm_inv_dur = [v / max_inv for v in inv_dur]

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
    # Higher weight on critical path (remaining_work) as it directly impacts makespan
    # Moderate weight on SPT to avoid long operations blocking machines
    # Lower weight on remaining ops as a tiebreaker
    w1 = 0.55  # remaining work (critical path) - dominant factor
    w2 = 0.30  # shortest processing time (soft-min)
    w3 = 0.15  # remaining ops urgency

    scores = []
    for i in range(n):
        s = w1 * norm_rw[i] + w2 * norm_inv_dur[i] + w3 * norm_ro[i]
        scores.append(s)

    best_idx = max(range(n), key=lambda i: scores[i])
    return int(best_idx)
# EVOLVE-BLOCK-END

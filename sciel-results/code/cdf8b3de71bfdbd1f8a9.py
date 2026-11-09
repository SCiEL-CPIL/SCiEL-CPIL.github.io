"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a refined composite heuristic:
      1. Primary: Maximize remaining_work (Longest Remaining Processing Time).
      2. Secondary: Minimize min_duration (Shortest Processing Time on best machine).
      3. Tertiary: Maximize remaining_ops count.
      4. Machine selection: Choose the machine that minimizes (current_time + duration),
         i.e., the machine that would be free earliest to start this operation.
    
    Returns: (index, machine) tuple to explicitly select the best machine,
             or just index if only one option.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    best_machine_indices = []
    best_durations = []
    remaining_works = []
    remaining_ops_counts = []
    job_ids = []
    
    for i, op in enumerate(ready_operations):
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        job = op.get("job", 0)
        
        # Find the machine with shortest duration for this operation
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_dur = machine_options[best_machine]
        else:
            best_machine = None
            best_dur = 0
        
        best_machine_indices.append(best_machine)
        best_durations.append(best_dur)
        remaining_works.append(mw)
        remaining_ops_counts.append(ro)
        job_ids.append(job)
    
    # Normalize values for composite scoring
    # Refined weights: emphasize remaining_work (critical path) most heavily,
    # with strong emphasis on processing time (shorter is better for parallelism),
    # and moderate emphasis on remaining ops.
    
    max_rw = max(remaining_works) if remaining_works else 1
    min_dur = min(best_durations) if best_durations else 1
    max_dur = max(best_durations) if best_durations else 1
    max_ro = max(remaining_ops_counts) if remaining_ops_counts else 1
    
    # Use a smooth log-sum-exp based soft-min to select machines and compute
    # a smooth duration score. This avoids the discontinuity of hard min.
    import math
    
    scores = []
    for i in range(n):
        # Normalize remaining work: higher is better
        rw_score = remaining_works[i] / max_rw if max_rw > 0 else 0
        
        # For duration scoring, use a smooth approach:
        # Consider all machine options with a soft-min weighting
        machine_options = ready_operations[i].get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            # Soft-min: exp(-durations) weighted average gives preference to shorter
            # but smooths over the hard minimum
            max_neg = max(-d for d in durations)
            exp_sum = sum(math.exp(-d - max_neg) for d in durations)
            if exp_sum > 0:
                # Smoothed effective duration (lower is better)
                soft_min_dur = -math.log(exp_sum) - max_neg
            else:
                soft_min_dur = min(durations)
        else:
            soft_min_dur = 0
        
        # Normalize soft_min_dur: shorter is better
        if max_dur > min_dur:
            dur_norm = (soft_min_dur - min_dur) / (max_dur - min_dur)
        else:
            dur_norm = 0.0
        # Clamp to [0, 1]
        dur_norm = max(0.0, min(1.0, dur_norm))
        # Smooth penalty: quadratic emphasizes short durations
        dur_score = 1.0 - dur_norm ** 1.5
        
        # Normalize remaining ops: higher is better
        ro_score = remaining_ops_counts[i] / max_ro if max_ro > 0 else 0
        
        # Composite score with refined weights
        # Emphasize remaining_work most, then duration, then ops count
        score = 0.55 * rw_score + 0.35 * dur_score + 0.10 * ro_score
        scores.append(score)
    
    # Find the index with maximum score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    # Machine selection: use soft-min to pick the machine that balances
    # duration and potential future availability. For simplicity, pick the
    # machine with minimum duration (which is what best_machine_indices already does).
    # But refine: if there are close durations, prefer the one that might
    # have less future contention (heuristic: lower machine index as tie-breaker
    # for determinism, or we could track machine loads - but we don't have that info).
    # Keep the min-duration machine selection as it's a good local heuristic.
    
    if best_machine_indices[best_idx] is not None:
        return (best_idx, best_machine_indices[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END

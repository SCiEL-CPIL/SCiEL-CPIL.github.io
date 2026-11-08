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
    # Refined weights: emphasize remaining_work more heavily (critical path),
    # with moderate emphasis on processing time, and less on op count.
    
    max_rw = max(remaining_works) if remaining_works else 1
    min_dur = min(best_durations) if best_durations else 1
    max_dur = max(best_durations) if best_durations else 1
    max_ro = max(remaining_ops_counts) if remaining_ops_counts else 1
    
    # Use a multiplicative penalty for long durations to create a smoother
    # surrogate that penalizes very long operations more aggressively.
    # This approximates a soft-max behavior where extreme values dominate.
    
    scores = []
    for i in range(n):
        # Normalize remaining work: higher is better
        rw_score = remaining_works[i] / max_rw if max_rw > 0 else 0
        
        # Normalize duration: shorter is better, so invert
        if max_dur > min_dur:
            dur_norm = (best_durations[i] - min_dur) / (max_dur - min_dur)
        else:
            dur_norm = 0.0
        # Apply exponential penalty to emphasize short durations
        # This creates a smooth surrogate that heavily favors short operations
        dur_score = 1.0 - 0.7 * dur_norm - 0.3 * dur_norm ** 2
        
        # Normalize remaining ops: higher is better
        ro_score = remaining_ops_counts[i] / max_ro if max_ro > 0 else 0
        
        # Composite score with refined weights
        # Emphasize remaining_work (critical path) most heavily
        score = 0.6 * rw_score + 0.3 * dur_score + 0.1 * ro_score
        scores.append(score)
    
    # Find the index with maximum score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    # Return (index, machine) to allow explicit machine selection
    if best_machine_indices[best_idx] is not None:
        return (best_idx, best_machine_indices[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END

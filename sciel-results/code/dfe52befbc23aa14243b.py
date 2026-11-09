"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE_SECONDS = 140.0  # safety margin under 150s wall-clock kill
_START_TIME = _time.monotonic()


def _elapsed():
    return _time.monotonic() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: Multi-factor heuristic combining:
      1. Remaining work (critical path length proxy) - higher is worse, so we prefer ops from jobs with more remaining work to keep them progressing
      2. Minimum processing time among available machines (shortest job first on the chosen machine to reduce makespan)
      3. Machine flexibility / load balancing: prefer machines that are less loaded (fewer remaining ops assigned or less total work)
      4. Number of machine options (more options = more flexible, schedule later; fewer options = more constrained, schedule now)
    
    We use a weighted score where lower score = better choice.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, return a quick heuristic
    if _elapsed() > _DEADLINE_SECONDS:
        # Fallback: just pick the one with max remaining work
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i]["remaining_work"]))

    n = len(ready_operations)
    
    # Pre-compute some global stats for machine load balancing
    # Track total remaining work per machine across all ready operations
    machine_work = {}
    machine_op_count = {}
    
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_work[m] = machine_work.get(m, 0) + dur
            machine_op_count[m] = machine_op_count.get(m, 0) + 1
    
    # Find the max machine work for normalization
    max_machine_work = max(machine_work.values()) if machine_work else 1.0
    
    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Factor 1: Remaining work - jobs with more remaining work should be prioritized
        # Higher remaining work = more urgent, so we want to start these sooner
        # Score contribution: negative (we want to minimize score, so high remaining work = low score)
        work_score = -remaining_work
        
        # Factor 2: Minimum processing time - shorter operations reduce makespan
        min_duration = min(op["machine_options"].values())
        # Lower min duration = better, so positive contribution (higher score = worse)
        duration_score = min_duration
        
        # Factor 3: Machine flexibility - operations with fewer machine options are more constrained
        num_options = len(op["machine_options"])
        # Fewer options = more constrained = should schedule sooner
        # So fewer options = lower score (better)
        flexibility_score = -num_options  # negative: fewer options = more negative = better
        
        # Factor 4: Remaining ops count - jobs with more remaining ops need attention
        # But this is somewhat captured by remaining_work
        # Use a small weight
        ops_score = remaining_ops * 0.1
        
        # Factor 5: Load balancing - prefer machines with less total work
        # For each machine option, compute the load; take the min load machine
        min_machine_load = min(
            machine_work.get(m, 0) / max_machine_work for m in op["machine_options"]
        )
        # Lower load = better
        load_score = min_machine_load
        
        # Combine with weights
        # Tuned weights: work is most important, then duration, then flexibility
        score = (
            1.0 * work_score +       # primary: remaining work
            0.3 * duration_score +    # secondary: min processing time
            0.2 * flexibility_score + # tertiary: machine flexibility
            0.05 * ops_score +        # quaternary: remaining ops
            0.15 * load_score         # quinary: machine load balancing
        )
        
        scores.append(score)
    
    # Pick the operation with the lowest score (best)
    best_idx = min(range(n), key=lambda i: scores[i])
    
    return int(best_idx)
# EVOLVE-BLOCK-END

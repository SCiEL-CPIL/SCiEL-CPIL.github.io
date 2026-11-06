"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def dispatch(ready_operations, time):
    """
    Dispatch rule using Longest Remaining Work (LRW) heuristic.
    
    Heuristic:
    Priority = Total Remaining Work for the Job
    
    Rationale:
    - In Job-Shop Scheduling, minimizing makespan is often achieved by prioritizing
      jobs with the largest total remaining processing time (Longest Processing Time, LPT).
      This reduces the risk of "tail effects" where a long job is left for last,
      extending the overall makespan.
    - `remaining_work` represents the sum of processing times for all remaining
      operations in the job. Prioritizing higher values ensures that long jobs
      start earlier.
    - Tie-breaking: If remaining work is equal, we prioritize the operation with the
      shortest minimum processing time (Shortest Processing Time, SPT) to keep
      machines busy with quick tasks if the long job isn't immediately critical,
      or simply to break ties deterministically towards efficiency.
    
    Machine Selection:
    - We select the machine with the minimum processing duration for the chosen
      operation. This minimizes the time the job occupies the machine, potentially
      allowing the next operation in the job sequence to start sooner if the machine
      is the bottleneck, or freeing the machine for other jobs.
    """
    if not ready_operations:
        return 0

    best_idx = 0
    best_score = float('-inf')
    best_tiebreak = float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            continue

        # Get the minimum processing time among all eligible machines
        min_duration = min(machine_options.values())
        
        # Primary Score: Total remaining work for the job (Higher is better)
        # If 'remaining_work' is not present or 0, fall back to remaining_ops
        remaining_work = op.get("remaining_work", 0)
        if remaining_work == 0:
            # Fallback: estimate remaining work as count of ops * min_duration?
            # Or just use remaining_ops as a proxy for "size"
            remaining_work = op.get("remaining_ops", 1) * min_duration

        # Secondary Score (Tie-breaker): Shorter current operation is better (Lower is better)
        # We want to minimize min_duration, so we negate it for "higher is better" logic
        # or just compare separately.
        
        if remaining_work > best_score:
            best_score = remaining_work
            best_tiebreak = min_duration
            best_idx = i
            best_machine = min(machine_options, key=machine_options.get)
        elif remaining_work == best_score:
            if min_duration < best_tiebreak:
                best_tiebreak = min_duration
                best_idx = i
                best_machine = min(machine_options, key=machine_options.get)

    return (int(best_idx), best_machine)
# EVOLVE-BLOCK-END

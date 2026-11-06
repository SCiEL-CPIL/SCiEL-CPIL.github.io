"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Explicit deadline guard: 150s wall-clock limit, use 145s for safety
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling using greedy bottleneck elimination.
    
    Strategy:
    - Select the ready operation with the maximum estimated critical path length
      (remaining_ops * min_duration) to prioritize jobs that dominate makespan
    - Assign to the machine with the minimum duration (lowest load proxy) to
      prevent early congestion on any single machine
    
    This shifts focus from ratio-based heuristics to directly targeting the
    longest remaining processing chain and spreading load across machines.
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for idx, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        min_duration = min(durations)
        
        # Critical path estimate: number of remaining operations times the
        # minimum possible duration per operation. This approximates the
        # longest chain of work that must still be completed for this job.
        # Jobs with more remaining operations and/or longer minimum durations
        # are bottlenecks that will dominate the makespan if not addressed.
        critical_path = remaining_ops * min_duration
        
        # Secondary urgency: total remaining work to break ties when
        # critical paths are similar. Jobs with more total work are
        # inherently more impactful on makespan.
        urgency = critical_path + 0.1 * remaining_work
        
        # Cost: use the minimum duration as a proxy for how quickly this
        # operation can be completed. Lower cost = faster completion =
        # less time the machine is occupied.
        cost = min_duration + 1e-9
        
        # Score: prioritize by urgency/cost ratio. High urgency (long
        # critical path) and low cost (fast machine) get highest priority.
        score = urgency / cost
        
        if score > best_score:
            best_score = score
            best_idx = idx
            # Choose the machine with the minimum duration to minimize
            # occupation time and prevent congestion. This is a simple
            # and effective load-balancing heuristic.
            best_machine = machines[durations.index(min_duration)]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END

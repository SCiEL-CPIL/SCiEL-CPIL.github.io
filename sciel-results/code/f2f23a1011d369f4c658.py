"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Dispatch rule based on maximizing minimum slack time across jobs and machines.
    
    The idea is to model this as a constrained scheduling problem where we want to
    avoid creating bottlenecks. We prioritize operations that have the least slack,
    i.e., those that are most constrained by their latest possible start times.
    
    For each ready operation, we estimate its 'slack' based on:
    1. The remaining work on its job
    2. The remaining work on the machines it can use
    3. The duration of the operation itself
    
    We select the operation that, when scheduled, best preserves the minimum slack
    across all active jobs and machines.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Collect information about remaining work per job and per machine
    job_remaining_work = defaultdict(float)
    machine_remaining_work = defaultdict(float)
    
    for op in ready_operations:
        job = op["job"]
        # Add the minimum duration across machine options as a proxy for this op's contribution
        min_duration = min(op["machine_options"].values()) if op["machine_options"] else 0
        job_remaining_work[job] += min_duration
        
        for machine, duration in op["machine_options"].items():
            machine_remaining_work[machine] += duration
    
    # For each ready operation, compute a slack-based score
    # Lower slack = more urgent = higher priority
    scores = []
    for i, op in enumerate(ready_operations):
        job = op["job"]
        options = op["machine_options"]
        
        if not options:
            scores.append(0)
            continue
        
        # For each machine option, compute the slack if we assign this op to that machine
        # Slack is estimated as: remaining work on job + remaining work on machine - 2*duration
        # We want to pick the machine that maximizes this slack (least constraining)
        # But for dispatch priority, we want operations with LOWEST slack to be scheduled first
        
        best_slack = float('inf')
        for machine, duration in options.items():
            # Estimate the bottleneck: the minimum of job remaining and machine remaining
            # after this operation
            job_after = job_remaining_work[job] - min(options.values())
            machine_after = machine_remaining_work[machine] - duration
            
            # Slack: how much "buffer" we have. Low slack means tight constraint.
            # We use the minimum of the remaining works as a proxy for constraint
            slack = min(job_after, machine_after)
            if slack < best_slack:
                best_slack = slack
        
        # If no machine options or all invalid, use a large value (low priority)
        if best_slack == float('inf'):
            best_slack = 0
        
        # We want to prioritize operations with LOW slack (more constrained)
        # So we negate or use negative slack for argmax
        scores.append(-best_slack)
    
    # Fallback: if all scores are the same or zero, use remaining_work as tiebreaker
    if max(scores) == 0:
        scores = [op["remaining_work"] for op in ready_operations]
    
    # Select the operation with the highest score (most constrained / most urgent)
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END

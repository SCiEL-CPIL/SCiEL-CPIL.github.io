# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dynamic bottleneck identification strategy.
    
    For each ready operation, we consider assigning it to its least-loaded available machine
    (estimated via duration). We then compute the minimum slack time across all currently ready
    operations after hypothetically assigning that operation. The goal is to select the operation
    whose assignment maximizes the minimum slack time, thereby preventing machine starvation
    and reducing idle gaps.
    
    Slack time for a ready operation is estimated as:
      slack = remaining_work - estimated_time_to_complete_if_started_now
    
    We approximate the impact of starting an operation on other operations' slack by considering
    that starting an operation on a machine reduces the availability of that machine, which could
    increase the effective remaining work for other operations that need that machine.
    
    A simpler, more computationally tractable approach:
    - For each ready operation i, find the machine m_i that minimizes duration (least loaded / fastest).
    - Estimate the "load" added to machine m_i by this operation.
    - For each other ready operation j, estimate how much its slack is affected if it also wants m_i.
    - Compute the minimum slack across all ready operations under this hypothetical assignment.
    - Select the operation i that maximizes this minimum slack.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    if n == 1:
        return 0
    
    # Precompute for each operation:
    # - best machine (min duration)
    # - best duration
    # - remaining_work
    # - remaining_ops count
    best_machines = []
    best_durations = []
    remaining_works = []
    remaining_ops_counts = []
    all_machines = set()
    
    for op in ready_operations:
        mo = op["machine_options"]
        best_m = min(mo.keys(), key=lambda m: mo[m])
        best_machines.append(best_m)
        best_durations.append(mo[best_m])
        remaining_works.append(op["remaining_work"])
        remaining_ops_counts.append(op["remaining_ops"])
        all_machines.update(mo.keys())
    
    # For each candidate operation i, simulate assigning it to its best machine
    # and compute the minimum slack across all ready operations.
    
    # Improved slack estimation:
    # The slack of an operation represents how much "breathing room" it has.
    # We define slack_j as:
    #   slack_j = remaining_works[j] / max(remaining_ops_counts[j], 1) - contention_penalty_j
    #
    # The first term is the average remaining work per operation, which gives a sense
    # of how much work still needs to be done.
    #
    # The contention penalty captures the delay that operation j might face if it
    # competes for the same machine as the candidate operation i.
    #
    # If operation j prefers the same machine as operation i (m_i), then j might
    # have to wait for i to finish, adding d_i to its effective completion time.
    # This reduces its slack by d_i.
    #
    # Additionally, we consider the second-best machine for j. If j's second-best
    # machine is also m_i, the contention is even worse.
    
    # Precompute second-best machines for each operation
    second_best_machines = []
    for op in ready_operations:
        mo = op["machine_options"]
        machines_sorted = sorted(mo.keys(), key=lambda m: mo[m])
        if len(machines_sorted) > 1:
            second_best_machines.append(machines_sorted[1])
        else:
            second_best_machines.append(None)
    
    best_score = -float('inf')
    best_index = 0
    
    for i in range(n):
        m_i = best_machines[i]
        d_i = best_durations[i]
        
        min_slack = float('inf')
        for j in range(n):
            if j == i:
                # Operation i is being started now, so its remaining work after this op is:
                # remaining_works[i] - d_i
                # Its average remaining work per op after this is:
                # (remaining_works[i] - d_i) / max(remaining_ops_counts[i] - 1, 1)
                rem_ops_after = max(remaining_ops_counts[i] - 1, 1)
                slack_j = (remaining_works[i] - d_i) / rem_ops_after
            else:
                # Operation j is not started yet.
                # Base slack is average remaining work per op
                base_slack = remaining_works[j] / max(remaining_ops_counts[j], 1)
                
                # Contention penalty
                penalty = 0.0
                if best_machines[j] == m_i:
                    # j's best machine is taken by i, so j must wait d_i
                    penalty += d_i
                    # If j's second-best machine is also m_i (unlikely but possible),
                    # the penalty is even worse, but let's just use d_i for now
                elif second_best_machines[j] == m_i:
                    # j's second-best machine is taken by i, so if j can't use its best
                    # machine for some reason, it would face a delay. We add a smaller penalty.
                    penalty += 0.5 * d_i
                
                slack_j = base_slack - penalty
            
            if slack_j < min_slack:
                min_slack = slack_j
        
        if min_slack > best_score:
            best_score = min_slack
            best_index = i
    
    return int(best_index)
# EVOLVE-BLOCK-END

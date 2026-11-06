"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_sgs(instance, priority_scores):
    """
    Simulate a serial schedule generation scheme given priority scores.
    Returns the makespan (total duration of the schedule).
    """
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (R,)
    R = len(capacities)

    # Predecessors: list of lists, preds[i] = list of activity indices that must precede i
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(N)]
    else:
        # Ensure it's a list of lists
        if isinstance(preds, (list, tuple)):
            preds = [list(p) if isinstance(p, (list, tuple)) else [p] for p in preds]
        else:
            preds = [list(p) for p in preds]

    # Build successor list for faster completion checking
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)

    # Count of uncompleted predecessors for each activity
    unmet_preds = [len(preds[i]) for i in range(N)]

    # Track which activities are completed
    completed = [False] * N

    # Resource usage over time: we'll track events
    # For simplicity, we simulate event-by-event.
    # At each step, find eligible activities (all predecessors completed, resources available at current time)
    # and schedule the one with highest priority.

    # We'll use a timeline approach:
    # - current_time: the time at which we are making decisions
    # - active_activities: activities currently running, with their end times
    # - resource_usage: current usage of each resource

    current_time = 0.0
    active = {}  # activity_index -> end_time
    resource_usage = np.zeros(R, dtype=float)
    scheduled_count = 0

    # Precompute priority order (highest priority first)
    # We need to repeatedly find the highest-priority eligible activity
    # For efficiency, we can pre-sort and filter, but N=30 is small

    while scheduled_count < N:
        # Find eligible activities: not completed, not active, all preds met, resources available
        eligible = []
        for i in range(N):
            if completed[i] or i in active:
                continue
            if unmet_preds[i] > 0:
                continue
            # Check resource availability
            can_start = True
            for r in range(R):
                if resource_usage[r] + demands[i, r] > capacities[r] + 1e-9:
                    can_start = False
                    break
            if can_start:
                eligible.append(i)

        if not eligible:
            # No activity can start now; advance time to next completion
            if not active:
                # This should not happen if there's a valid schedule
                break
            next_time = min(end_time for end_time in active.values())
            current_time = next_time
            # Complete all activities that end at this time
            to_complete = [i for i, et in active.items() if et <= current_time + 1e-9]
            for i in to_complete:
                end_time = active.pop(i)
                completed[i] = True
                resource_usage -= demands[i]
                for s in succs[i]:
                    unmet_preds[s] -= 1
                scheduled_count += 1
            continue

        # Among eligible, pick the one with highest priority score
        best_i = max(eligible, key=lambda i: priority_scores[i])

        # Schedule best_i at current_time
        start_time = current_time
        end_time = start_time + durations[best_i]
        active[best_i] = end_time
        resource_usage += demands[best_i]
        # Note: best_i is now active, not yet completed
        # It's not "completed" until it finishes

        # After scheduling, we might be able to start more activities at the same time
        # But in serial SGS, we schedule one at a time. However, if resources allow,
        # we can start multiple at the same time. Let's continue the loop to check
        # if more activities can start at the same current_time.
        # Actually, in serial SGS, we pick one, then re-evaluate. So we continue.
        # But we don't increment scheduled_count yet because the activity hasn't completed.
        # Wait, scheduled_count should count activities that have been started.
        # Let me reconsider: scheduled_count should be the number of activities started.
        # Let me fix this.

    # The above logic has an issue. Let me rewrite more carefully.
    pass


def _simulate_sgs_v2(instance, priority_scores):
    """
    Correct simulation of serial SGS.
    """
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (R,)
    R = len(capacities)

    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(N)]
    else:
        if isinstance(preds, (list, tuple)):
            preds = [list(p) if isinstance(p, (list, tuple)) else [p] for p in preds]
        else:
            preds = [list(p) for p in preds]

    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)

    unmet_preds = [len(preds[i]) for i in range(N)]
    completed = [False] * N
    started = [False] * N

    current_time = 0.0
    active = {}  # activity_index -> end_time
    resource_usage = np.zeros(R, dtype=float)
    started_count = 0
    makespan = 0.0

    while started_count < N:
        # Find eligible activities
        eligible = []
        for i in range(N):
            if started[i]:
                continue
            if unmet_preds[i] > 0:
                continue
            can_start = True
            for r in range(R):
                if resource_usage[r] + demands[i, r] > capacities[r] + 1e-9:
                    can_start = False
                    break
            if can_start:
                eligible.append(i)

        if not eligible:
            if not active:
                break  # Should not happen for valid instances
            next_time = min(active.values())
            current_time = next_time
            to_complete = [i for i, et in active.items() if et <= current_time + 1e-9]
            for i in to_complete:
                active.pop(i)
                completed[i] = True
                resource_usage -= demands[i]
                for s in succs[i]:
                    unmet_preds[s] -= 1
            continue

        # Pick highest priority
        best_i = max(eligible, key=lambda i: priority_scores[i])

        # Start best_i
        start_time = current_time
        end_time = start_time + durations[best_i]
        active[best_i] = end_time
        resource_usage += demands[best_i]
        started[best_i] = True
        started_count += 1

        makespan = max(makespan, end_time)

    # Complete all remaining active activities
    if active:
        makespan = max(makespan, max(active.values()))

    return makespan


def priority(instance):
    """
    Returns a priority score for each activity.
    Uses a combination of critical path length and resource intensity.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    R = len(capacacities) if False else len(capacities)

    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(N)]
    else:
        if isinstance(preds, (list, tuple)):
            preds = [list(p) if isinstance(p, (list, tuple)) else [p] for p in preds]
        else:
            preds = [list(p) for p in preds]

    # Compute longest path from each activity to the end (critical path length)
    # This is the longest path starting at activity i (including i's duration)
    # to any terminal activity.
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)

    # Compute longest path from i to end
    # Use topological order or memoized DFS
    import sys
    sys.setrecursionlimit(10000)

    def longest_path(i):
        if i in memo:
            return memo[i]
        if not succs[i]:
            memo[i] = durations[i]
            return durations[i]
        max_succ = 0.0
        for s in succs[i]:
            max_succ = max(max_succ, longest_path(s))
        memo[i] = durations[i] + max_succ
        return memo[i]

    memo = {}
    for i in range(N):
        longest_path(i)
    cp_lengths = np.array([memo[i] for i in range(N)])

    # Resource intensity: for each activity, how much of the bottleneck resource it uses
    # Normalize demands by capacities
    # intensity[i] = sum over r of (demands[i,r] / capacities[r]) if capacities[r] > 0
    intensity = np.zeros(N)
    for r in range(R):
        if capacities[r] > 1e-9:
            intensity += demands[:, r] / capacities[r]
        else:
            if demands[:, r].any():
                intensity += demands[:, r] * 1000  # High penalty if capacity is 0 but demand > 0

    # Combine: high CP length and high intensity -> high priority
    # Normalize both to [0, 1]
    cp_min, cp_max = cp_lengths.min(), cp_lengths.max()
    if cp_max - cp_min > 1e-9:
        cp_norm = (cp_lengths - cp_min) / (cp_max - cp_min)
    else:
        cp_norm = np.ones(N)

    int_min, int_max = intensity.min(), intensity.max()
    if int_max - int_min > 1e-9:
        int_norm = (intensity - int_min) / (int_max - int_min)
    else:
        int_norm = np.ones(N)

    # Weighted combination
    alpha = 0.7  # Weight for critical path
    beta = 0.3   # Weight for intensity
    scores = alpha * cp_norm + beta * int_norm

    # Higher score = higher priority
    return scores
# EVOLVE-BLOCK-END

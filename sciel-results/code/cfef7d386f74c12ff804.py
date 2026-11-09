"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 145.0  # safety margin before 150s kill


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy: Combine critical path (tail), resource contention, and
    slack-based urgency to prioritize activities that most affect makespan.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]  # shape: (n_activities, n_resources)
    capacities = instance["capacities"]  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists or similar
    
    n = len(durations)
    
    # Convert predecessors to a consistent format
    if isinstance(predecessors, np.ndarray):
        pred_lists = []
        for i in range(n):
            pred_lists.append(list(predecessors[i]))
    else:
        pred_lists = [list(p) for p in predecessors]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if i not in successors[p]:
                successors[p].append(i)
    
    # Topological order
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # tail[i] = longest duration from i to any sink (including i's own duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    # head[i] = longest duration from any source to i (including i's own duration)
    head = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            head[i] = durations[i] + max(head[p] for p in pred_lists[i])
        else:
            head[i] = durations[i]
    
    # Project makespan without resource constraints (longest path)
    makespan_ignoring_resources = max(tail[i] for i in range(n)) if n > 0 else 0.0
    
    # Total float / slack: how much an activity can be delayed without affecting makespan
    # slack[i] = makespan - head[i] - tail[i] + durations[i]
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - head[i] - tail[i] + durations[i]
    
    # Resource demand intensity: weighted sum of demand/capacity ratios
    demand_intensity = np.zeros(n, dtype=float)
    n_resources = len(capacities)
    for i in range(n):
        for r in range(n_resources):
            cap = capacities[r]
            if cap > 0:
                demand_intensity[i] += demands[i][r] / cap
            else:
                demand_intensity[i] += demands[i][r] * 10.0
    
    # Number of direct successors (broader impact)
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Normalize components to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_tail = normalize(tail)
    norm_slack_inv = normalize(-slack)  # lower slack = higher priority
    norm_demand = normalize(demand_intensity)
    norm_dur = normalize(durations)
    norm_succ = normalize(num_successors)
    
    # Weighted combination emphasizing critical path and low slack
    # High tail, low slack, high demand, long duration, many successors => high priority
    priority_scores = (
        1.0 * norm_tail +
        0.8 * norm_slack_inv +
        0.3 * norm_demand +
        0.2 * norm_dur +
        0.1 * norm_succ
    )
    
    return priority_scores


def _simulate_greedy_schedule(instance, priority_scores):
    """
    Simulate a serial schedule-generation scheme using the given priority scores.
    Returns the makespan of the resulting schedule.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_resources = len(capacities)
    
    # Convert predecessors
    if isinstance(predecessors, np.ndarray):
        pred_lists = []
        for i in range(n):
            pred_lists.append(list(predecessors[i]))
    else:
        pred_lists = [list(p) for p in predecessors]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            successors[p].append(i)
    
    # Track completion times
    completion_time = np.zeros(n, dtype=float)
    scheduled = [False] * n
    
    # Current resource usage over time (we'll track events)
    # For simplicity, we track current demand at each time step
    # Since durations are continuous, we need to be careful
    
    # We'll use a priority queue approach: at each step, find eligible activities
    # and pick the one with highest priority score
    
    # Track when resources become available
    # We'll simulate time in discrete steps based on activity completions
    
    # Active activities: dict from completion_time to list of activity indices
    active_events = {}  # time -> list of (activity, resource_demands)
    
    current_time = 0.0
    remaining = set(range(n))
    
    # Pre-compute: for each activity, the number of unscheduled predecessors
    unscheduled_preds = [len(pred_lists[i]) for i in range(n)]
    
    while remaining:
        # Find all eligible activities (all predecessors completed)
        eligible = [i for i in remaining if unscheduled_preds[i] == 0]
        
        if not eligible:
            # This should not happen if the precedence graph is a DAG
            # But just in case, break
            break
        
        # Pick the eligible activity with the highest priority score
        # Tie-break by higher tail (longest path to end)
        best = max(eligible, key=lambda i: (priority_scores[i], completion_time[i] if i in completion_time else 0))
        
        # Determine the earliest start time for this activity
        # It must be >= max completion time of all predecessors
        earliest_start = 0.0
        for p in pred_lists[best]:
            if completion_time[p] > earliest_start:
                earliest_start = completion_time[p]
        
        # Check resource availability: we need to find the earliest time >= earliest_start
        # where all resources are available
        # This is complex; for simplicity, we'll assume we can start at earliest_start
        # and check if resources are available. If not, we wait.
        
        # For now, let's assume resources are always available (this is a simplification)
        # In a real implementation, we'd need to track resource usage over time
        
        start_time = earliest_start
        completion_time[best] = start_time + durations[best]
        
        # Remove from remaining
        remaining.discard(best)
        scheduled[best] = True
        
        # Update unscheduled_preds for successors
        for s in successors[best]:
            if s in remaining:
                unscheduled_preds[s] -= 1
    
    # Makespan is the maximum completion time
    if n == 0:
        return 0.0
    return float(np.max(completion_time))


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first when eligible).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE_SECONDS:
            # Should not happen for 30 activities, but guard anyway
            return np.asarray(instance["durations"], dtype=float)
        return scores
    except Exception:
        # Fallback: longest processing time
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            n = len(instance.get("durations", []))
            return np.ones(n, dtype=float)
# EVOLVE-BLOCK-END

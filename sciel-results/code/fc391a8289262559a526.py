"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_bottom_level(instance):
    """Compute bottom level (latest finish time from each node to project end).
    
    Bottom level of a node = max over successors (duration[succ] + bottom_level[succ]).
    Sinks have bottom_level = their own duration.
    This is the critical-path distance to the end, used as a priority (higher = more urgent).
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    predecessors = [[] for _ in range(n)]
    for i in range(n):
        preds = instance["predecessors"][i] if i < len(instance["predecessors"]) else []
        for p in preds:
            predecessors[i].append(p)
            successors[p].append(i)
    
    # Compute bottom level via reverse topological order (Kahn's algorithm reversed)
    # First, get topological order
    in_degree = [len(p) for p in predecessors]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If graph has cycles, fall back to all nodes in order
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Reverse topological order for bottom level computation
    bottom_level = np.zeros(n)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            val = durations[succ] + bottom_level[succ]
            if val > max_succ:
                max_succ = val
        bottom_level[node] = durations[node] + max_succ
    
    return bottom_level, successors, predecessors


def _compute_resource_pressure(instance):
    """Compute a smooth resource pressure metric for each activity.
    
    For each activity, estimate how much it competes for scarce resources
    relative to its successors. Uses a soft-max over resource contention.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    
    if demands.ndim == 1:
        demands = demands.reshape(n, -1)
    
    num_resources = capacities.shape[0] if capacities.ndim > 0 else demands.shape[1]
    
    # For each activity, compute the "scarcity ratio" per resource:
    # demand / capacity. Then use a smooth aggregation.
    # Higher scarcity means the activity is more likely to cause resource conflicts.
    # We weight by duration to capture the total resource-time footprint.
    
    # scarcity[i, r] = demands[i, r] / capacities[r]
    with np.errstate(divide='ignore', invalid='ignore'):
        scarcity = demands / capacities[np.newaxis, :]
    
    # Clip to avoid extreme values
    scarcity = np.clip(scarcity, 0, 10.0)
    
    # Soft-max over resources for each activity (smooth max)
    # Use log-sum-exp trick for numerical stability
    scale = 10.0  # temperature for soft-max
    exp_scaled = np.exp(scarcity * scale)
    log_sum_exp = np.log(np.sum(exp_scaled, axis=1, keepdims=True))
    soft_max_scarcity = (1.0 / scale) * (np.log(np.sum(exp_scaled, axis=1)) - 0)
    # Actually let's use a simpler approach: weighted sum with emphasis on peak
    # Peak scarcity * duration gives resource-time pressure
    peak_scarcity = np.max(scarcity, axis=1)
    mean_scarcity = np.mean(scarcity, axis=1)
    
    # Resource pressure = duration * (peak_scarcity * 0.6 + mean_scarcity * 0.4)
    resource_pressure = durations * (0.6 * peak_scarcity + 0.4 * mean_scarcity)
    
    return resource_pressure


def priority(instance):
    """Compute priority scores for activities in a resource-constrained project.
    
    Combines:
    1. Bottom level (critical path to end) - primary signal
    2. Resource pressure - secondary signal to avoid resource bottlenecks
    3. Duration - tertiary tiebreaker
    
    Higher score = higher priority (scheduled earlier in serial SGS).
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    
    # Compute bottom level (critical path distance to end)
    bottom_level, successors, predecessors = _compute_bottom_level(instance)
    
    if time.time() > deadline:
        return bottom_level.copy()
    
    # Compute resource pressure
    resource_pressure = _compute_resource_pressure(instance)
    
    if time.time() > deadline:
        return bottom_level.copy()
    
    # Normalize components to comparable scales
    bl_range = bottom_level.max() - bottom_level.min() if bottom_level.max() > bottom_level.min() else 1.0
    rp_range = resource_pressure.max() - resource_pressure.min() if resource_pressure.max() > resource_pressure.min() else 1.0
    dur_range = durations.max() - durations.min() if durations.max() > durations.min() else 1.0
    
    bl_norm = (bottom_level - bottom_level.min()) / bl_range
    rp_norm = (resource_pressure - resource_pressure.min()) / rp_range
    dur_norm = (durations - durations.min()) / dur_range
    
    # Combine: bottom level dominates, resource pressure is a penalty for
    # activities that are resource-heavy but not on the critical path
    # We want to schedule critical-path activities first, but avoid
    # scheduling resource-heavy non-critical activities that would block
    # critical successors.
    
    # Strategy: primary priority is bottom level. Among activities with
    # similar bottom levels, prefer those with lower resource pressure
    # (less likely to cause resource conflicts for successors).
    
    # Weighted combination with emphasis on bottom level
    score = 0.70 * bl_norm + 0.20 * (1.0 - rp_norm) + 0.10 * dur_norm
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_successors(predecessors):
    """Compute successors from predecessors."""
    n = len(predecessors)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    return successors


def _critical_path_length(instance):
    """Compute longest path from each activity to the end (backward pass)."""
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    successors = _compute_successors(predecessors)
    
    # Topological order (reverse)
    in_degree = [len(preds) for preds in predecessors]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Backward pass: cp_length[i] = max duration from i to end
    cp_length = np.zeros(n, dtype=float)
    for i in reversed(topo):
        max_succ = 0.0
        for succ in successors[i]:
            if cp_length[succ] > max_succ:
                max_succ = cp_length[succ]
        cp_length[i] = durations[i] + max_succ
    
    return cp_length


def _total_demand_weight(instance):
    """Compute total resource demand weight for each activity."""
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, m)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (m,)
    
    # Normalize demands by capacities to get relative scarcity
    if capacities.size > 0 and np.any(capacities > 0):
        normalized = demands / np.maximum(capacities, 1e-9)
    else:
        normalized = demands
    
    # Sum across resources
    total_weight = np.sum(normalized, axis=1)
    return total_weight


def _resource_pressure(instance, activity_idx):
    """Compute how constrained an activity is based on resource demands."""
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    if capacities.size == 0:
        return 0.0
    
    d = demands[activity_idx]
    # Fraction of capacity consumed
    frac = d / np.maximum(capacities, 1e-9)
    return np.sum(frac)


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first).
    
    Uses a combination of:
    1. Critical path length (backward pass) - activities on the critical path get high priority
    2. Resource demand pressure - activities consuming more resources get priority
    3. Duration - longer activities get some priority
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute critical path lengths (backward pass)
    cp_length = _critical_path_length(instance)
    
    # Compute resource demand weights
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    if demands.ndim == 1:
        # Single resource case
        if capacities.size > 0 and capacities[0] > 0:
            resource_weight = demands / capacities[0]
        else:
            resource_weight = np.zeros(n, dtype=float)
    else:
        if capacities.size > 0 and np.any(capacities > 0):
            normalized = demands / np.maximum(capacities, 1e-9)
        else:
            normalized = demands
        resource_weight = np.sum(normalized, axis=1)
    
    # Normalize each component to [0, 1] range
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(cp_length)
    dur_norm = normalize(durations)
    res_norm = normalize(resource_weight)
    
    # Combine: critical path is most important, then resource pressure, then duration
    # We want to prioritize activities that are on the critical path AND consume scarce resources
    # This helps avoid resource conflicts that would delay critical activities
    
    score = 0.5 * cp_norm + 0.3 * res_norm + 0.2 * dur_norm
    
    # Add a small bonus for activities with high resource pressure relative to their position
    # to break ties in favor of more resource-intensive tasks
    if np.any(resource_weight > 0):
        score += 0.1 * res_norm
    
    # Ensure no negative scores
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

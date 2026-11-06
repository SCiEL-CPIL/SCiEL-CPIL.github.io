"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graphs(n, predecessors):
    """Build successor lists and in-degree counts from predecessors."""
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            succ[p].append(i)
            indeg[i] += 1
    return succ, indeg


def _earliest_start(n, durations, predecessors, succ, indeg_template):
    """Compute earliest start times via forward pass."""
    es = [0.0] * n
    indeg = list(indeg_template)
    queue = [i for i in range(n) if indeg[i] == 0]
    processed = 0
    while queue:
        i = queue.pop()
        processed += 1
        for j in succ[i]:
            new_es = es[i] + durations[i]
            if new_es > es[j]:
                es[j] = new_es
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    return es


def _latest_finish(n, durations, predecessors, succ, indeg_template, horizon):
    """Compute latest finish times via backward pass."""
    lf = [float(horizon)] * n
    # Reverse in-degree: number of successors
    r_indeg = [0] * n
    r_succ = [[] for _ in range(n)]
    for i in range(n):
        for j in succ[i]:
            r_indeg[i] += 1
            r_succ[j].append(i)
    
    r_indeg_copy = list(r_indeg)
    queue = [i for i in range(n) if r_indeg_copy[i] == 0]
    while queue:
        i = queue.pop()
        for j in r_succ[i]:
            new_lf = lf[i] - durations[i]
            if new_lf < lf[j]:
                lf[j] = new_lf
            r_indeg_copy[j] -= 1
            if r_indeg_copy[j] == 0:
                queue.append(j)
    return lf


def priority(instance):
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    r = len(capacities)
    
    # Build graph structures
    succ, indeg_template = _build_graphs(n, predecessors)
    
    # Compute earliest start times
    es = _earliest_start(n, durations, predecessors, succ, indeg_template)
    
    # Compute a rough horizon (sum of all durations as upper bound)
    horizon = float(np.sum(durations))
    
    # Compute latest finish times
    lf = _latest_finish(n, durations, predecessors, succ, indeg_template, horizon)
    
    # Compute slack (total float) for each activity
    # slack[i] = lf[i] - es[i] - durations[i]
    slack = np.array([lf[i] - es[i] - durations[i] for i in range(n)], dtype=float)
    
    # Ensure non-negative slack (numerical safety)
    slack = np.maximum(slack, 0.0)
    
    # Compute resource pressure: for each activity, sum of (demand/capacity) across resources
    # This measures how "scarce" the resources are that this activity uses
    # Normalize: higher ratio means more constrained resource
    resource_pressure = np.zeros(n, dtype=float)
    for i in range(n):
        for k in range(r):
            if capacities[k] > 0:
                resource_pressure[i] += demands[i, k] / capacities[k]
    
    # Compute successor count (number of direct successors) as a proxy for "importance"
    succ_count = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Compute a criticality weight: inverse slack (higher for critical activities)
    # Use a smooth inverse to avoid division by zero
    eps = 1e-6
    inv_slack = 1.0 / (slack + eps)
    
    # Combine: prioritize activities that are
    # 1. On the critical path (low slack) -> high inv_slack
    # 2. Use scarce resources -> high resource_pressure
    # 3. Have many successors (more downstream impact) -> high succ_count
    # 4. Have long duration (blocks resources longer) -> high duration
    
    # Normalize each component to [0, 1] range for stable combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_inv_slack = normalize(inv_slack)
    norm_pressure = normalize(resource_pressure)
    norm_succ = normalize(succ_count)
    norm_dur = normalize(durations)
    
    # Weighted combination
    # Emphasize criticality (inverse slack) most, then resource pressure
    w_slack = 0.45
    w_pressure = 0.30
    w_succ = 0.15
    w_dur = 0.10
    
    scores = (w_slack * norm_inv_slack + 
              w_pressure * norm_pressure + 
              w_succ * norm_succ + 
              w_dur * norm_dur)
    
    # Check time
    if time.time() > deadline:
        return scores
    
    # Secondary refinement: consider "resource contention" more carefully
    # For each activity, estimate how much it delays others by occupying resources
    # Use a simple proxy: for each resource k, the fraction of capacity used
    # Combined with duration gives a "resource-blocking" measure
    blocking = np.zeros(n, dtype=float)
    for i in range(n):
        for k in range(r):
            if capacities[k] > 0:
                blocking[i] += (demands[i, k] / capacities[k]) * durations[i]
    
    norm_blocking = normalize(blocking)
    
    # Blend in blocking measure
    final_scores = 0.7 * scores + 0.3 * norm_blocking
    
    if time.time() > deadline:
        return final_scores
    
    return final_scores
# EVOLVE-BLOCK-END

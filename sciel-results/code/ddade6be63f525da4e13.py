"""
Combined program for Resource-Constrained Project Scheduling.

Strategy:
- Uses the multiplicative "Resource-Weighted Critical Path" approach from the current program,
  which outperformed the additive approach.
- Incorporates the "Latest Start Time" (LST) / Slack heuristic from the directive,
  prioritizing activities with the least scheduling flexibility (lowest slack).
- Uses Log-Sum-Exp (LSE) for resource pressure (from Inspiration 1) to capture peak congestion
  more robustly than simple sums, highlighting bottleneck resources.
- The priority score is: (BL / (BL + Slack)) * (1 + NormalizedLSEPressure) + DurationTieBreaker.
- This captures that critical activities (high BL) with low flexibility (low Slack) that also
  use resources with high "peak load" (high LSE pressure) should be scheduled first.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

def priority(instance):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    # 1. Parse Instance Data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Normalize predecessors to lists of ints
    pred_lists = []
    for i in range(n):
        p = predecessors[i]
        if p is None:
            p = []
        if isinstance(p, (list, tuple, np.ndarray)):
            p = [int(x) for x in p]
        else:
            p = [int(p)]
        # Filter out self and out-of-range
        p = [x for x in p if 0 <= x < n and x != i]
        pred_lists.append(p)

    # Build successor lists
    succ_lists = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_lists[i]:
            succ_lists[p].append(i)
            in_degree[i] += 1

    # 2. Topological Sort (Kahn's Algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    # Use a copy of in_degree for the sort
    temp_in_degree = in_degree.copy()
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in succ_lists[node]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)
                
    # Handle potential cycles (fallback to original order if cycle detected, though RCPSP should be acyclic)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(list(remaining))

    # 3. Calculate Bottom Level (BL) and Top Level (TL)
    # BL[i] = durations[i] + max(BL[s] for s in successors[i])
    # TL[i] = durations[i] + max(TL[p] for p in predecessors[i])
    BL = np.zeros(n, dtype=float)
    TL = np.zeros(n, dtype=float)
    
    # Forward pass for TL
    for node in topo_order:
        if pred_lists[node]:
            max_pred_tl = max(TL[p] for p in pred_lists[node])
        else:
            max_pred_tl = 0.0
        TL[node] = durations[node] + max_pred_tl
        
    # Backward pass for BL
    for node in reversed(topo_order):
        if succ_lists[node]:
            max_succ_bl = max(BL[s] for s in succ_lists[node])
        else:
            max_succ_bl = 0.0
        BL[node] = durations[node] + max_succ_bl

    # 4. Calculate Slack (Total Float)
    # Project duration is the max BL (or max TL, they should be equal in a valid DAG)
    project_duration = float(np.max(BL)) if n > 0 else 0.0
    
    # Slack[i] = (project_duration - BL[i] - TL[i]) + durations[i]
    # Alternatively: Slack = LS - ES. 
    # ES = TL[i] - durations[i]
    # LS = (project_duration - BL[i]) - durations[i]
    # Slack = LS - ES = (project_duration - BL[i] - durations[i]) - (TL[i] - durations[i])
    # Slack = project_duration - BL[i] - TL[i] + durations[i]
    
    # Note: If we define TL as earliest start + duration, and BL as latest start + duration?
    # Let's stick to standard definitions:
    # TL[i] = Earliest Finish Time (EF)
    # BL[i] = Latest Finish Time (LF)
    # Slack = LF - EF
    
    slack = project_duration - BL - TL + durations
    
    # Ensure non-negative slack (numerical stability)
    slack = np.maximum(slack, 0.0)
    
    # 5. Calculate Resource Pressure using Log-Sum-Exp (Fluid Flow / Peak Congestion)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_pressure = np.zeros(n, dtype=float)
    
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # Handle shape variations
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, 1)
        if demands_arr.shape[0] != n:
            # Fallback if shape mismatch
            demands_arr = np.zeros((n, 1), dtype=float)
            if demands_arr.size > 0:
                demands_arr[0, 0] = 1.0
        
        R = demands_arr.shape[1]
        
        if capacities_arr.size == 1:
            caps = np.full(R, float(capacities_arr[0]), dtype=float)
        else:
            caps = capacities_arr.reshape(-1)
            if caps.size < R:
                tmp = np.ones(R, dtype=float)
                tmp[:caps.size] = caps
                caps = tmp
            else:
                caps = caps[:R]
        
        # Ensure no division by zero
        caps = np.maximum(caps, 1e-9)
        
        # Calculate normalized load per resource
        norm_load = demands_arr / caps[np.newaxis, :]  # (n, R)
        
        # Log-Sum-Exp for smooth max
        max_load = np.max(norm_load, axis=1, keepdims=True)
        
        # Handle case where max_load is 0 (no demand)
        # If max_load is 0, LSE should be 0.
        # exp(norm_load - max_load) is safe.
        
        lse_pressure = max_load[:, 0] + np.log(np.sum(np.exp(norm_load - max_load), axis=1))
        
        # Normalize the LSE pressure to [0, 1]
        lp_max = float(np.max(lse_pressure)) if n > 0 else 1.0
        if lp_max > 1e-12:
            resource_pressure = lse_pressure / lp_max
        else:
            resource_pressure = np.zeros(n, dtype=float)

    # 6. Combine Scores
    # Base score: Inverse slack normalized by BL to create a "criticality" factor.
    # If Slack is 0, factor is 1. If Slack is high, factor is lower.
    # We want High BL and Low Slack to result in High Score.
    
    # Use a soft inverse: BL / (BL + Slack)
    # This is 1 when Slack=0, and approaches 0 when Slack >> BL.
    # It is always in (0, 1].
    
    # Add a small epsilon to avoid division by zero if BL is 0 (which shouldn't happen for non-zero durations)
    bl_safe = np.maximum(BL, 1e-9)
    slack_factor = bl_safe / (bl_safe + slack)
    
    # Multiplicative interaction with Resource Pressure
    # High Slack Factor (critical) * High Pressure (scarce resource) = Very High Priority
    score = slack_factor * (1.0 + resource_pressure)
    
    # Tie-breaker: Slightly favor longer durations to break ties
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max > 1e-12:
        dur_norm = durations / dur_max
        score += 0.01 * dur_norm
    else:
        score += 0.0

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return np.asarray(score, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END

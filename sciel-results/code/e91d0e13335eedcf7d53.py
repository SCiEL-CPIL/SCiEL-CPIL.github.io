"""
Combined program for Resource-Constrained Project Scheduling.

Strategy:
- Uses the multiplicative "Resource-Weighted Critical Path" approach as the base.
- Incorporates the "fluid-flow bottleneck-pressure" concept from Inspiration 1,
  specifically using log-sum-exp (LSE) to create a smooth surrogate for the
  peak resource contention. This is more robust than simple sums as it
  highlights the most constrained resource dimension.
- Retains the robust data parsing and topological sorting from the current program.
- The priority score is: BL * (1 + NormalizedLSEPressure).
- This captures that activities on the critical path (high BL) that also use
  resources with high "peak load" (high LSE pressure) should be scheduled first.
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

    # 3. Calculate Bottom Level (BL)
    # BL[i] = durations[i] + max(BL[s] for s in successors[i])
    BL = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succ_lists[node]:
            max_succ_bl = max(BL[s] for s in succ_lists[node])
        else:
            max_succ_bl = 0.0
        BL[node] = durations[node] + max_succ_bl

    # 4. Calculate Resource Pressure using Log-Sum-Exp (Fluid Flow / Peak Congestion)
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
        
        # Calculate "load area" per resource: demand * duration
        # This represents the total "volume" of resource usage.
        # Dividing by capacity gives a normalized load.
        # Using LSE over resources gives a smooth max-like measure of peak congestion.
        
        # We want to identify activities that are "heavy" on the bottleneck resources.
        # Let's compute the normalized demand per resource: d_ij / c_j
        # Then we want a scalar score that is high if any d_ij/c_j is large.
        # LSE is a good soft-max.
        
        norm_load = demands_arr / caps[np.newaxis, :]  # (n, R)
        
        # To make LSE stable and meaningful, we might scale it.
        # Standard LSE: max(x) + log(sum(exp(x - max(x))))
        # If norm_load is large, LSE is large.
        
        # Let's use the log-sum-exp formulation directly on the normalized loads.
        # This effectively computes a "soft maximum" of the resource usage ratios.
        
        # Stability: subtract max
        max_load = np.max(norm_load, axis=1, keepdims=True)
        # Avoid log(0) if all loads are 0
        # If max_load is 0, LSE should be 0 (or small).
        
        # exp(norm_load - max_load) is safe since max_load is the max.
        # Sum is at least 1 (from the max term).
        
        # However, if R is large, sum(exp(...)) can be large.
        # The LSE value approximates max(norm_load) + log(R) if all are equal.
        
        lse_pressure = max_load[:, 0] + np.log(np.sum(np.exp(norm_load - max_load), axis=1))
        
        # Normalize the LSE pressure to [0, 1] to balance with BL.
        # BL is in time units. LSE is in "normalized load" units.
        # We want to add a multiplier (1 + pressure) to BL.
        
        lp_max = float(np.max(lse_pressure)) if n > 0 else 1.0
        if lp_max > 1e-12:
            resource_pressure = lse_pressure / lp_max
        else:
            resource_pressure = np.zeros(n, dtype=float)

    # 5. Combine Scores
    # Multiplicative interaction: High BL * High Pressure = Very High Priority
    # This captures that critical activities using scarce resources are most urgent.
    score = BL * (1.0 + resource_pressure)
    
    # Tie-breaker: Slightly favor longer durations to break ties in BL/Pressure
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max > 1e-12:
        dur_norm = durations / dur_max
        score += 0.01 * dur_norm
    else:
        score += 0.0

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return score

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END

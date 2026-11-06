"""
RCPSP Heuristic: Lookahead Contention Priority
This strategy simulates the serial schedule generation process using a baseline
heuristic (Longest Processing Time) to identify activities that cause resource
contention. It combines this with critical path analysis (Total Float) to score
activities. Activities that are critical AND consume scarce resources are given
the highest priority, as scheduling them early resolves bottlenecks that would
otherwise delay the project.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Returns a priority score for each activity.
    Higher score = higher priority.
    
    Strategy:
    1. Parse instance data (durations, predecessors, demands, capacities).
    2. Perform a CPM (Critical Path Method) forward and backward pass to get
       Total Float (TF) for each activity.
    3. Simulate a serial schedule using a baseline heuristic (LPT) to estimate
       'resource contention'. We track how much each activity is delayed by
       resources compared to its earliest start time (ES).
    4. Calculate a 'contention score' based on resource demand intensity.
    5. Combine TF (criticality), contention (bottleneck potential), and duration.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Handle potential missing keys or different formats
    preds_raw = instance.get("predecessors", instance.get("preds", [[] for _ in range(n)]))
    demands_raw = instance.get("demands", instance.get("resource_demands", None))
    capacities_raw = instance.get("capacities", instance.get("resource_capacities", None))
    
    # Normalize predecessors to list of lists of ints
    preds = [list(p) if p else [] for p in preds_raw]
    if len(preds) < n:
        preds = [[] for _ in range(n)]
        for i in range(n):
            if i < len(preds_raw) and preds_raw[i]:
                preds[i] = list(preds_raw[i])
                
    # Build successors
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                succs[i].append(j)
                
    # Handle resources
    if demands_raw is not None:
        try:
            demands = np.asarray(demands_raw, dtype=float)
            if demands.ndim == 1:
                demands = demands.reshape(-1, 1)
        except:
            demands = np.zeros((n, 1))
    else:
        demands = np.zeros((n, 1))
        
    if capacities_raw is not None:
        try:
            capacities = np.asarray(capacities_raw, dtype=float)
            if capacities.ndim == 0:
                capacities = np.array([capacities])
        except:
            capacities = np.ones(demands.shape[1])
    else:
        capacities = np.ones(demands.shape[1])
        
    m_resources = demands.shape[1]
    if len(capacities) < m_resources:
        capacities = np.pad(capacities, (0, m_resources - len(capacities)), mode='constant')
        
    # Topological Sort
    in_degree = np.array([len(preds[i]) for i in range(n)], dtype=int)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        # Cycle detected or invalid graph, fallback to duration
        return durations.copy()
        
    # 1. CPM Forward Pass (Unlimited Resources)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in topo_order:
        es[u] = 0
        for p in preds[u]:
            if ef[p] > es[u]:
                es[u] = ef[p]
        ef[u] = es[u] + durations[u]
        
    makespan_cpm = np.max(ef)
    
    # 2. CPM Backward Pass
    lf = np.zeros(n)
    ls = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) > 0:
            lf[u] = np.min([ls[v] for v in succs[u]])
        else:
            lf[u] = makespan_cpm
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es  # Total Float
    
    # 3. Resource Contention Simulation
    # We simulate the serial SGS using LPT (Longest Processing Time) as a baseline.
    # We want to identify which activities are "tight" in terms of resources.
    # A simple proxy for contention is the ratio of demand to capacity.
    # However, a better proxy is how much an activity delays others.
    # Since we can't easily run a full look-ahead search for each activity,
    # we use a "Resource Pressure" score:
    # Pressure = sum(demand_i / capacity_i) for each resource i.
    # This identifies tasks that consume a large fraction of available resources.
    
    if m_resources > 0 and np.all(capacities > 0):
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        # Normalize pressure per resource to avoid one dominant resource
        # Actually, we want the total load.
        pressure = np.sum(demands / safe_cap, axis=1)
        max_pressure = np.max(pressure)
        if max_pressure > 0:
            pressure = pressure / max_pressure
        else:
            pressure = np.zeros(n)
    else:
        pressure = np.zeros(n)
        
    # 4. Combine Scores
    # We want high priority for:
    # a) Low Total Float (Critical) -> High Score
    # b) High Resource Pressure (Bottleneck) -> High Score
    # c) Long Duration (LPT) -> High Score
    
    # Normalize TF to [0, 1] where 0 is critical (high priority)
    max_tf = np.max(tf) if np.max(tf) > 0 else 1.0
    crit_score = 1.0 - (tf / max_tf) # 1.0 for critical, 0.0 for max float
    
    # Normalize Duration
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_score = durations / max_dur
    
    # Weights:
    # Criticality is most important.
    # Pressure is secondary.
    # Duration is tertiary.
    
    # Adjust weights to emphasize criticality more strongly
    w_crit = 1.0
    w_press = 0.3
    w_dur = 0.1
    
    score = w_crit * crit_score + w_press * pressure + w_dur * dur_score
    
    # Optional: Time check
    if time.time() - start_time > DEADLINE:
        return score
        
    return score

# EVOLVE-BLOCK-END

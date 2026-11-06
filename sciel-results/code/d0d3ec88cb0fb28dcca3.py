"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled earlier).
    
    Strategy: Combine multiple heuristics:
    1. Resource-weighted inverse slack (proven to improve from 0.903 -> 1.046)
    2. Total float / backward pass slack
    3. Duration
    4. Resource scarcity (how constrained the resources are)
    5. Precedence depth (successor count)
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    
    # Parse resource demands and capacities
    # instance["demands"] is a list of lists: demands[i][r] = resource r demand for activity i
    # instance["capacities"] is a list of resource capacities
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    num_resources = len(capacities)
    
    # Parse predecessors: instance["predecessors"] is a list of lists
    predecessors = instance["predecessors"]
    
    # Build successor graph
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute resource scarcity: for each resource, how tight is it?
    # A resource is scarce if total demand is close to capacity
    total_demand = demands.sum(axis=0)  # shape (num_resources,)
    # Scarcity: 1 if total_demand > capacity, scaled
    scarcity = np.zeros(num_resources)
    for r in range(num_resources):
        if capacities[r] > 0:
            scarcity[r] = min(total_demand[r] / capacities[r], 2.0)
        else:
            scarcity[r] = 0.0
    
    # Resource pressure per activity: sum of demand_r * scarcity_r / capacity_r
    # Normalize by capacity to get relative pressure
    resource_pressure = np.zeros(n)
    for i in range(n):
        for r in range(num_resources):
            if capacities[r] > 0:
                resource_pressure[i] += demands[i, r] * scarcity[r] / capacities[r]
    
    # Forward pass: earliest start times (ignoring resources)
    est = np.zeros(n)
    # Topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
    
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Forward pass
    est = np.zeros(n)
    for node in topo_order:
        for pred in predecessors[node]:
            est[node] = max(est[node], est[pred] + durations[pred])
    
    # Project makespan (ignoring resources)
    makespan = est[-1] + durations[-1] if n > 0 else 0
    # More robust: max over all nodes
    if n > 0:
        makespan = max(est[i] + durations[i] for i in range(n))
    
    # Backward pass: latest start times
    lft = np.full(n, makespan)  # latest finish time
    # Process in reverse topological order
    for node in reversed(topo_order):
        for succ in successors[node]:
            lft[node] = min(lft[node], lft[succ] - durations[succ])
        lft[node] = max(lft[node], est[node])  # safety
    
    # Slack (total float)
    slack = lft - est - durations  # latest start - earliest start
    slack = np.maximum(slack, 0)
    
    # Inverse slack (higher = more critical)
    inv_slack = 1.0 / (slack + 1.0)
    
    # Successor count (how many activities depend on this one)
    succ_count = np.zeros(n)
    for i in range(n):
        succ_count[i] = len(successors[i])
    
    # Number of immediate successors
    num_succ = np.zeros(n)
    for i in range(n):
        num_succ[i] = len(successors[i])
    
    # Combine heuristics:
    # 1. Resource-weighted inverse slack (most important based on findings)
    # 2. Duration
    # 3. Inverse slack
    # 4. Successor count (activities with more successors should go first)
    
    # Normalize each component
    def normalize(x):
        x = np.asarray(x, dtype=float)
        mn, mx = x.min(), x.max()
        if mx - mn > 1e-10:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)
    
    # Resource-weighted inverse slack
    rw_inv_slack = resource_pressure * inv_slack
    rw_inv_slack_norm = normalize(rw_inv_slack)
    
    # Duration normalized
    dur_norm = normalize(durations)
    
    # Inverse slack normalized
    inv_slack_norm = normalize(inv_slack)
    
    # Successor count normalized
    succ_norm = normalize(succ_count)
    
    # Resource pressure normalized
    rp_norm = normalize(resource_pressure)
    
    # Weighted combination
    # Emphasize resource-weighted inverse slack (proven effective)
    weights = {
        'rw_inv_slack': 0.35,
        'duration': 0.20,
        'inv_slack': 0.20,
        'succ_count': 0.10,
        'resource_pressure': 0.15,
    }
    
    score = (weights['rw_inv_slack'] * rw_inv_slack_norm +
             weights['duration'] * dur_norm +
             weights['inv_slack'] * inv_slack_norm +
             weights['succ_count'] * succ_norm +
             weights['resource_pressure'] * rp_norm)
    
    return score
# EVOLVE-BLOCK-END

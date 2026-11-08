"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

_DEADLINE = 120.0  # seconds
_START_TIME = None


def _serial_schedule(durations, demands, capacities, predecessors, priority_scores):
    """
    Serial schedule generation scheme.
    durations: (n,) array of activity durations
    demands: (n, m) array of resource demands per activity per resource
    capacities: (m,) array of resource capacities
    predecessors: list of lists, predecessors[i] is list of indices
    priority_scores: (n,) array of priority scores (higher = earlier)
    Returns: makespan (total project duration)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = len(durations)
    m = len(capacities)
    
    # Build successor lists for quick checking
    in_degree = [0] * n
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            successors[p].append(i)
    
    # Track start times and completion
    start_times = [0] * n
    completed = [False] * n
    eligible = set()
    
    # Initialize eligible activities (no predecessors)
    for i in range(n):
        if in_degree[i] == 0:
            eligible.add(i)
    
    current_time = 0
    scheduled_count = 0
    
    resource_intervals = [[] for _ in range(m)]  # each entry: list of (start, end, amount)
    
    def get_resource_usage(r, t):
        """Get total usage of resource r at time t."""
        total = 0
        for (s, e, a) in resource_intervals[r]:
            if s <= t < e:
                total += a
        return total
    
    def can_start(activity, t):
        """Check if activity can start at time t given resource constraints."""
        for r in range(m):
            demand = demands[activity][r]
            if demand == 0:
                continue
            usage = get_resource_usage(r, t)
            if usage + demand > capacities[r]:
                return False
        return True
    
    def earliest_start(activity, t_min):
        """Find earliest time >= t_min when activity can start."""
        # Collect candidate times
        candidates = {t_min}
        for r in range(m):
            for (s, e, a) in resource_intervals[r]:
                if s > t_min:
                    candidates.add(s)
                if e > t_min:
                    candidates.add(e)
        
        candidate_times = sorted(candidates)
        
        for ct in candidate_times:
            if ct >= t_min and can_start(activity, ct):
                return ct
        
        # Fallback: extend beyond known intervals
        max_time = max(candidate_times) + 1000 if candidate_times else t_min + 1000
        t = t_min
        while t < max_time:
            if can_start(activity, t):
                return t
            next_bp = t + 1
            for r in range(m):
                for (s, e, a) in resource_intervals[r]:
                    if e > t and e < next_bp:
                        next_bp = e
            t = next_bp
        return t
    
    while scheduled_count < n:
        # Deadline check
        if time.time() - _START_TIME > _DEADLINE - 2.0:
            # Return a reasonable estimate
            if start_times:
                return max(st + durations[i] for i, st in enumerate(start_times))
            return 0
        
        # Find the highest priority eligible activity
        best_activity = None
        best_priority = -float('inf')
        for a in eligible:
            if priority_scores[a] > best_priority:
                best_priority = priority_scores[a]
                best_activity = a
        
        if best_activity is None:
            break
        
        # Find earliest start time
        st = earliest_start(best_activity, current_time)
        
        # Schedule it
        start_times[best_activity] = st
        et = st + durations[best_activity]
        
        # Add resource usage intervals
        for r in range(m):
            demand = demands[best_activity][r]
            if demand > 0:
                resource_intervals[r].append((st, et, demand))
        
        completed[best_activity] = True
        scheduled_count += 1
        eligible.discard(best_activity)
        
        if st > current_time:
            current_time = st
        
        # Mark successors as potentially eligible
        for succ in successors[best_activity]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                eligible.add(succ)
    
    # Calculate makespan
    if start_times:
        makespan = max(st + durations[i] for i, st in enumerate(start_times))
    else:
        makespan = 0
    
    return makespan


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demand"] if "demand" in instance else instance.get("demands", np.zeros((len(durations), 0))), dtype=float)
    capacities = np.asarray(instance["capacity"] if "capacity" in instance else instance.get("capacities", np.array([float('inf')])), dtype=float)
    predecessors = instance["predecessors"] if "predecessors" in instance else instance.get("predecessor_lists", [[] for _ in range(len(durations))])
    
    n = len(durations)
    m = len(capacities)
    
    # Ensure demands has shape (n, m)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
        capacities = capacities.reshape(1)
        m = 1
    
    if len(capacities) == 0:
        m = 0
        capacities = np.array([float('inf')])
        demands = np.zeros((n, 1))
    
    # Ensure m matches
    if demands.shape[1] < m:
        pad = np.zeros((n, m - demands.shape[1]))
        demands = np.hstack([demands, pad])
    elif demands.shape[1] > m:
        demands = demands[:, :m]
    
    # Compute successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute longest path from each activity to the end (tail)
    # Use iterative topological order to avoid recursion issues
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Topological order
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If there's a cycle, topo_order may not contain all nodes; handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes in arbitrary order
        in_topo = set(topo_order)
        for i in range(n):
            if i not in in_topo:
                topo_order.append(i)
    
    # Compute head (longest path from start to i, excluding i's duration)
    head = np.zeros(n)
    for node in topo_order:
        best = 0
        for p in predecessors[node]:
            path = head[p] + durations[p]
            if path > best:
                best = path
        head[node] = best
    
    # Compute tail (longest path from i to end, excluding i's duration)
    tail = np.zeros(n)
    for node in reversed(topo_order):
        best = 0
        for s in successors[node]:
            path = durations[s] + tail[s]
            if path > best:
                best = path
        tail[node] = best
    
    # Critical path length (project duration in unconstrained case)
    cp_length = max(head[i] + durations[i] + tail[i] for i in range(n))
    
    # Total float (slack) = cp_length - head[i] - durations[i] - tail[i]
    # Lower slack = more critical = higher priority
    total_float = cp_length - head - durations - tail
    total_float = np.maximum(total_float, 0)  # clamp to 0
    
    # Total resource demand per activity
    total_demand = demands.sum(axis=1)
    
    # Demand-to-capacity ratio (max across resources)
    demand_ratio = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0:
            ratio = demands[:, r] / capacities[r]
            demand_ratio = np.maximum(demand_ratio, ratio)
    
    # Successor count
    succ_count = np.array([len(s) for s in successors], dtype=float)
    pred_count = np.array([len(p) for p in predecessors], dtype=float)
    
    # Normalize each feature
    def normalize(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    # For total_float, we want LOWER float to have HIGHER priority
    # So we invert: 1 - normalize(total_float) gives high score for low float
    norm_float_inv = 1.0 - normalize(total_float)
    
    norm_duration = normalize(durations)
    norm_total_demand = normalize(total_demand)
    norm_tail = normalize(tail)
    norm_head = normalize(head)
    norm_demand_ratio = normalize(demand_ratio)
    norm_succ_count = normalize(succ_count)
    norm_pred_count = normalize(pred_count)
    
    # Composite score: weighted sum
    # Strong emphasis on total float (slack) as the primary signal
    # Activities with zero/low slack are on or near critical path
    w_float_inv = 0.45   # Strong weight on low slack (high priority)
    w_tail = 0.22        # Long tail means more downstream impact
    w_duration = 0.12    # Longer activities benefit from early scheduling
    w_total_demand = 0.08 # High resource demand - schedule early when resources free
    w_demand_ratio = 0.07 # High demand/capacity ratio - bottleneck
    w_head = 0.03        # Some weight on head (less important than tail)
    w_succ_count = 0.02  # More successors = more downstream impact
    w_pred_count = 0.01  # Fewer predecessors might mean more flexibility
    
    scores = (w_float_inv * norm_float_inv + 
              w_tail * norm_tail + 
              w_duration * norm_duration + 
              w_total_demand * norm_total_demand + 
              w_demand_ratio * norm_demand_ratio + 
              w_head * norm_head + 
              w_succ_count * norm_succ_count + 
              w_pred_count * norm_pred_count)
    
    return scores
# EVOLVE-BLOCK-END

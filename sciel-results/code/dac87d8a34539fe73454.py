"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache


def _sgs_makespan(durations, demands, capacities, preds, priority_scores):
    """
    Serial schedule generation scheme.
    Returns makespan for a given priority list.
    
    Parameters:
    - durations: array of activity durations
    - demands: dict {resource_idx: array of demands per activity}
    - capacities: dict {resource_idx: capacity}
    - preds: list of lists, preds[i] = list of predecessor indices for activity i
    - priority_scores: array of scores; higher score = higher priority
    """
    n = len(durations)
    # Sort activities by priority (descending)
    order = np.argsort(-priority_scores)
    
    # Track start times and completion times
    start = np.zeros(n, dtype=float)
    finish = np.zeros(n, dtype=float)
    
    # Track resource availability over time using event-based approach
    # For each resource, maintain a list of (time, amount_freed) events
    # But since we need to check at each start time, let's use a simpler approach:
    # For each resource, track current usage and schedule of releases
    
    # We'll use a timeline approach: for each resource, keep a dict of time -> amount released
    # Actually, let's just track for each resource a sorted list of (release_time, amount)
    # and compute available capacity at any time t
    
    # Simpler: for each resource, maintain a list of (start, end, amount) intervals of usage
    # and when scheduling, find earliest time when all resource demands can be met
    
    # For efficiency with 30 activities, we can do a straightforward check
    
    resource_usages = {r: [] for r in capacities}  # list of (start, end, amount)
    
    scheduled = set()
    
    for act in order:
        if act in scheduled:
            continue
        
        # Check if all predecessors are completed
        pred_done = all(p in scheduled for p in preds[act])
        if not pred_done:
            # In serial SGS, we process in priority order, but if predecessors aren't done,
            # we skip and come back. Actually, standard serial SGS processes activities
            # in priority order, and an activity is eligible only if all preds are done.
            # So we need to iterate through the priority list and only schedule eligible ones.
            
            # Let me restructure: iterate through priority list, and for each,
            # check eligibility. If not eligible, skip (will be revisited? No, in serial SGS,
            # we go through the list once, and only schedule if eligible at that point.)
            
            # Actually, the standard serial SGS is:
            # 1. Create a priority list
            # 2. For each activity in the priority list (in order):
            #    a. If all predecessors are already scheduled, schedule it at the earliest
            #       feasible time (considering resource constraints and predecessor finishes)
            #    b. If not all predecessors are scheduled, skip it (it will never be scheduled
            #       in this pass, which means the priority list must be a topological order)
            
            # So the priority list should be a topological order. Let me reconsider.
            
            # The task says "starts the highest-priority eligible activity". This suggests
            # a different scheme: repeatedly pick the highest-priority activity among those
            # that are eligible (all preds done) and schedule it at the earliest feasible time.
            
            # Let me implement that instead.
            pass
    
    # Let me redo this with the correct algorithm:
    # Repeatedly select the highest-priority eligible activity and schedule at earliest feasible time
    
    start = np.zeros(n, dtype=float)
    finish = np.zeros(n, dtype=float)
    scheduled = np.zeros(n, dtype=bool)
    resource_usages = {r: [] for r in capacities}
    
    remaining = set(range(n))
    
    while remaining:
        # Find eligible activities: all predecessors are scheduled
        eligible = []
        for act in remaining:
            if all(scheduled[p] for p in preds[act]):
                eligible.append(act)
        
        if not eligible:
            # Deadlock - shouldn't happen with valid input
            break
        
        # Pick the highest priority eligible activity
        best_act = max(eligible, key=lambda a: priority_scores[a])
        
        # Find earliest start time for this activity
        # Earliest start is at least the max finish time of predecessors
        earliest = 0.0
        for p in preds[best_act]:
            earliest = max(earliest, finish[p])
        
        # Now find the earliest time t >= earliest where all resource demands can be met
        t = earliest
        while True:
            feasible = True
            for r, cap in capacities.items():
                demand = demands[r][best_act]
                if demand == 0:
                    continue
                # Check if at time t, the available capacity is sufficient
                # Available = cap - sum of usages that overlap with [t, t+durations[best_act])
                end_time = t + durations[best_act]
                current_usage = 0
                for (s, e, amt) in resource_usages[r]:
                    # Check if [s, e) overlaps with [t, end_time)
                    if s < end_time and e > t:
                        current_usage += amt
                if current_usage + demand > cap:
                    feasible = False
                    # Find the earliest time when this resource frees up
                    # Look for the earliest end time among usages that overlap with [t, end_time)
                    min_end = float('inf')
                    for (s, e, amt) in resource_usages[r]:
                        if s < end_time and e > t:
                            min_end = min(min_end, e)
                    if min_end < float('inf'):
                        t = max(t, min_end)
                    else:
                        t += 1  # Fallback
                    break
            if feasible:
                break
            # Safety: prevent infinite loop
            if t > 100000:
                break
        
        # Schedule the activity
        start[best_act] = t
        finish[best_act] = t + durations[best_act]
        scheduled[best_act] = True
        remaining.discard(best_act)
        
        # Update resource usages
        for r, cap in capacities.items():
            demand = demands[r][best_act]
            if demand > 0:
                resource_usages[r].append((t, t + durations[best_act], demand))
    
    makespan = np.max(finish) if n > 0 else 0.0
    return makespan


def _compute_priority(instance, weights):
    """
    Compute priority scores based on weighted features.
    
    weights: dict with keys like 'duration', 'resource_demand', 'num_preds', 'num_successors', 
             'latest_start', 'slack', etc.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]
    capacities = instance["capacities"]
    preds = instance["predecessors"]
    
    n = len(durations)
    
    # Compute features
    # 1. Duration (longer first)
    f_duration = durations.copy()
    
    # 2. Total resource demand (sum of demands across all resources, normalized by capacity)
    f_resource = np.zeros(n)
    for r, cap in capacities.items():
        if cap > 0:
            f_resource += np.asarray(demands[r], dtype=float) / cap
        else:
            f_resource += np.asarray(demands[r], dtype=float)
    
    # 3. Number of predecessors
    f_num_preds = np.array([len(p) for p in preds], dtype=float)
    
    # 4. Number of successors (how many activities depend on this one)
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    f_num_succs = np.array([len(s) for s in succs], dtype=float)
    
    # 5. Latest start time (backward pass)
    # First, compute earliest finish times (forward pass)
    ef = np.zeros(n)
    # Topological order
    in_degree = np.array([len(p) for p in preds], dtype=int)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
    
    for node in topo_order:
        ef[node] = durations[node]
        for s in succs[node]:
            ef[s] = max(ef[s], ef[node] + durations[s])
    
    # Latest finish and latest start (backward pass)
    lf = np.zeros(n)
    ls = np.zeros(n)
    # Initialize lf for activities with no successors
    for i in range(n):
        if not succs[i]:
            lf[i] = ef[i]
        else:
            lf[i] = float('inf')
    
    # Process in reverse topological order
    for node in reversed(topo_order):
        if succs[node]:
            lf[node] = min(lf[s] for s in succs[node])
        else:
            lf[node] = ef[node]
        ls[node] = lf[node] - durations[node]
    
    f_latest_start = ls.copy()
    
    # 6. Slack = latest_start - earliest_start
    es = np.zeros(n)
    for node in topo_order:
        es[node] = 0 if not preds[node] else max(es[p] for p in preds[node])
        # Actually, es is determined by predecessors
    # Recompute es properly
    es = np.zeros(n)
    for node in topo_order:
        if preds[node]:
            es[node] = max(ef[p] for p in preds[node])
    f_slack = ls - es
    
    # Normalize features to [0, 1] range
    def normalize(f):
        if np.max(f) > np.min(f):
            return (f - np.min(f)) / (np.max(f) - np.min(f))
        return f
    
    f_duration_n = normalize(f_duration)
    f_resource_n = normalize(f_resource)
    f_num_preds_n = normalize(f_num_preds)
    f_num_succs_n = normalize(f_num_succs)
    f_latest_start_n = normalize(f_latest_start)
    f_slack_n = normalize(f_slack)
    
    # Compute weighted score
    score = np.zeros(n)
    if weights.get('duration', 0) != 0:
        score += weights['duration'] * f_duration_n
    if weights.get('resource_demand', 0) != 0:
        score += weights['resource_demand'] * f_resource_n
    if weights.get('num_preds', 0) != 0:
        score += weights['num_preds'] * f_num_preds_n
    if weights.get('num_succs', 0) != 0:
        score += weights['num_succs'] * f_num_succs_n
    if weights.get('latest_start', 0) != 0:
        score += weights['latest_start'] * f_latest_start_n
    if weights.get('slack', 0) != 0:
        score += weights['slack'] * f_slack_n
    
    return score


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]
    capacities = instance["capacities"]
    preds = instance["predecessors"]
    
    n = len(durations)
    
    # Define a set of candidate weight vectors
    # We'll search over combinations of features
    base_weights = [
        {'duration': 1.0, 'resource_demand': 0.0, 'num_preds': 0.0, 'num_succs': 0.0, 'latest_start': 0.0, 'slack': 0.0},
        {'duration': 0.0, 'resource_demand': 1.0, 'num_preds': 0.0, 'num_succs': 0.0, 'latest_start': 0.0, 'slack': 0.0},
        {'duration': 0.0, 'resource_demand': 0.0, 'num_preds': 0.0, 'num_succs': 1.0, 'latest_start': 0.0, 'slack': 0.0},
        {'duration': 0.0, 'resource_demand': 0.0, 'num_preds': 0.0, 'num_succs': 0.0, 'latest_start': 1.0, 'slack': 0.0},
        {'duration': 0.0, 'resource_demand': 0.0, 'num_preds': 0.0, 'num_succs': 0.0, 'latest_start': 0.0, 'slack': 1.0},
        {'duration': 1.0, 'resource_demand': 1.0, 'num_preds': 0.0, 'num_succs': 0.0, 'latest_start': 0.0, 'slack': 0.0},
        {'duration': 1.0, 'resource_demand': 0.0, 'num_preds': 0.0, 'num_succs': 1.0, 'latest_start': 0.0, 'slack': 0.0},
        {'duration': 0.5, 'resource_demand': 0.5, 'num_preds': 0.5, 'num_succs': 0.5, 'latest_start': 0.5, 'slack': 0.5},
        {'duration': 1.0, 'resource_demand': 0.5, 'num_preds': 0.0, 'num_succs': 0.5, 'latest_start': 0.5, 'slack': 0.0},
        {'duration': 0.5, 'resource_demand': 0.5, 'num_preds': 0.0, 'num_succs': 1.0, 'latest_start': 0.5, 'slack': 0.5},
    ]
    
    best_score = None
    best_makespan = float('inf')
    
    # Evaluate base weights
    for w in base_weights:
        if time.time() > deadline:
            break
        scores = _compute_priority(instance, w)
        makespan = _sgs_makespan(durations, demands, capacities, preds, scores)
        if makespan < best_makespan:
            best_makespan = makespan
            best_score = scores
    
    # If we have time, do local search around the best solution
    if best_score is not None and time.time() < deadline - 10:
        # Perturb the best weight vector
        best_w = None
        for w in base_weights:
            s = _compute_priority(instance, w)
            if np.allclose(s, best_score):
                best_w = dict(w)
                break
        
        if best_w is None:
            # Find closest weight vector
            min_dist = float('inf')
            for w in base_weights:
                s = _compute_priority(instance, w)
                dist = np.sum((s - best_score) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_w = dict(w)
        
        # Local search
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Randomly perturb one weight
            perturbed_w = dict(best_w)
            keys = list(perturbed_w.keys())
            key = np.random.choice(keys)
            perturbed_w[key] = max(0, perturbed_w[key] + np.random.uniform(-0.5, 0.5))
            
            scores = _compute_priority(instance, perturbed_w)
            makespan = _sgs_makespan(durations, demands, capacities, preds, scores)
            
            if makespan < best_makespan:
                best_makespan = makespan
                best_score = scores
                best_w = perturbed_w
    
    if best_score is not None:
        return best_score
    
    # Fallback: longest duration first
    return durations
# EVOLVE-BLOCK-END

"""RCPSP priority via local search with optimized SGS simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: 
    1. Compute features (Slack, Duration, Resource Criticality).
    2. Initialize priority based on Minimum Slack (strong heuristic).
    3. Use Hill Climbing / Simulated Annealing to refine priority scores
       by swapping or perturbing scores, evaluated via a fast SGS simulation.
    """
    start_time = time.time()
    DEADLINE = 118.0

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # --- Extract instance components ---
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]

    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break
    if demands is None:
        demands = np.zeros((n, 1))

    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists if not provided
    if succs is None:
        succs = [set() for _ in range(n)]
        for j in range(n):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n:
                    succs[idx].add(j)
        succs = [list(s) for s in succs]

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
            idx = int(i)
            if 0 <= idx < n:
                in_degree[j] += 1

    # Use a simple list as queue for topo sort
    q = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass ---
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]
        for v in succs[u]:
            if LS[u] < LF[v]:
                LF[v] = LS[u]

    slack = LS - ES

    # --- Feature computation ---
    # Normalize slack to [0,1]
    min_slack = np.min(slack)
    max_slack = np.max(slack)
    if max_slack > min_slack:
        norm_slack = (slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.zeros(n)
    
    # Priority: Higher is better. Low slack = High priority.
    # Base priority: 1 - norm_slack
    base_priority = 1.0 - norm_slack

    # --- Fast SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS with O(N^2 * R) complexity per step? 
        Actually, we iterate N steps. Each step finds eligible (O(N)), picks best (O(N)), 
        finds start time (O(N) candidates * O(N) overlap check * R).
        Total: O(N^3 * R). With N=30, R=4, 30^3*4 = 1,080,000 ops. 
        This is fast enough for ~1000 simulations in 120s? 
        1000 * 1M = 1B ops. Python might be too slow.
        We need to optimize the overlap check.
        """
        if time.time() - start_time > DEADLINE:
            return float('inf')

        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        pred_set = [set() for _ in range(n)]
        for j in range(n):
            p = preds[j]
            if isinstance(p, (list, tuple, set, np.ndarray)):
                for i in p:
                    idx = int(i)
                    if 0 <= idx < n:
                        pred_set[j].add(idx)

        remaining = set(range(n))
        
        # To speed up overlap check, we can maintain a list of scheduled activities.
        # scheduled_list = [] # list of (idx, start, end)
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                return float('inf')

            # Find eligible
            eligible = []
            for j in remaining:
                # Check preds
                is_eligible = True
                for p in pred_set[j]:
                    if start_t[p] == np.inf:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(j)

            if not eligible:
                # Fallback: should not happen if topo order is correct and no cycles
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break

            # Pick highest priority
            best = max(eligible, key=lambda x: prio_scores[x])

            # Earliest start from precedence
            es_best = 0.0
            for p in pred_set[best]:
                cand = start_t[p] + durations[p]
                if cand > es_best:
                    es_best = cand

            # Candidate times: es_best and all finish times of scheduled activities >= es_best
            cand_times = [es_best]
            for k in range(n):
                if finish_t[k] != np.inf and finish_t[k] >= es_best:
                    cand_times.append(finish_t[k])
            
            # Remove duplicates and sort
            cand_times = sorted(set(cand_times))

            best_start = es_best
            found = False
            for t in cand_times:
                if t < es_best:
                    continue
                end_t = t + durations[best]
                
                # Check resource feasibility
                feasible = True
                # For each resource r, check if sum of demands of active activities + demand[best] <= cap[r]
                # Active in [t, end_t)
                # We can check at each "event" point in [t, end_t)
                # Event points: t, end_t, and start/finish times of scheduled activities that overlap [t, end_t)
                
                # Optimization: Just check the peak usage.
                # Since N is small, we can just iterate through scheduled activities and sum demands if they overlap.
                # But we need to ensure the SUM does not exceed capacity at ANY point.
                # The sum is constant between event points.
                # So we only need to check at the start of each overlapping activity and t.
                
                check_points = set()
                check_points.add(t)
                check_points.add(end_t)
                
                for k in range(n):
                    if start_t[k] != np.inf:
                        # If k overlaps [t, end_t)
                        if start_t[k] < end_t and finish_t[k] > t:
                            # Add start and end of k if within [t, end_t)
                            if t <= start_t[k] < end_t:
                                check_points.add(start_t[k])
                            if t < finish_t[k] <= end_t:
                                check_points.add(finish_t[k])
                
                check_points = sorted(check_points)
                
                for i in range(len(check_points) - 1):
                    tp = check_points[i]
                    tp_next = check_points[i+1]
                    if tp >= end_t or tp_next <= t:
                        continue
                    
                    # Sum demands of activities active in [tp, tp_next)
                    # Active if start <= tp and finish >= tp_next
                    active_demand = np.zeros(R)
                    for k in range(n):
                        if start_t[k] != np.inf:
                            if start_t[k] <= tp and finish_t[k] >= tp_next:
                                active_demand += demands[k]
                    
                    # Add best activity
                    active_demand += demands[best]
                    
                    if np.any(active_demand > capacities + 1e-9):
                        feasible = False
                        break
                
                if feasible:
                    best_start = t
                    found = True
                    break

            if not found:
                best_start = es_best

            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            remaining.discard(best)

        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Search Strategy ---
    # Start with Min Slack
    best_scores = base_priority.copy()
    best_ms = simulate_sgs(best_scores)
    
    # Hill Climbing / Random Search
    # We will perturb the scores directly.
    # Since scores are relative, we can add noise or swap values.
    
    current_scores = best_scores.copy()
    current_ms = best_ms
    
    # Temperature for Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    max_iterations = 5000
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Generate neighbor
        # Option 1: Swap two random activities' scores
        # Option 2: Add noise to one activity's score
        # Option 3: Linear combination of base features with random weights
        
        # Let's try swapping scores of two activities that are "close" in priority or random.
        if random.random() < 0.5:
            # Swap
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], current_scores[i1]
        else:
            # Perturb one
            i1 = random.randint(0, n-1)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.1)
            # Ensure scores remain in a reasonable range? 
            # Actually, only relative order matters.
        
        # Evaluate neighbor
        neighbor_ms = simulate_sgs(neighbor_scores)
        
        if neighbor_ms is float('inf'):
            continue
            
        # Accept if better, or with probability exp(-Delta/T)
        delta = neighbor_ms - current_ms
        if delta < 0:
            # Better
            current_scores = neighbor_scores
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_scores = current_scores.copy()
        else:
            # Worse, accept with prob
            if random.random() < np.exp(-delta / T):
                current_scores = neighbor_scores
                current_ms = neighbor_ms
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_scores
# EVOLVE-BLOCK-END

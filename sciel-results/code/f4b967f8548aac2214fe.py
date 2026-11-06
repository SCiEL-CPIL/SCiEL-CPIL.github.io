"""RCPSP priority via MCTS-informed local search with fast SGS simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: 
    1. Compute rich features (Slack, Duration, Resource Criticality, Contention).
    2. Initialize priority based on a strong composite heuristic.
    3. Use MCTS to explore the decision space of which activity to schedule next
       from the initial eligible set, using fast SGS rollouts as the value function.
    4. Refine the resulting priority ordering via local search (hill climbing / SA).
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

    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)

    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
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
    def normalize(x):
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    # Resource tightness
    resource_tightness = np.zeros(R)
    for r in range(R):
        total_demand = float(np.sum(demands[:, r]))
        if total_demand > 0:
            resource_tightness[r] = capacities[r] / total_demand
        else:
            resource_tightness[r] = 1.0

    max_demand_per_res = np.max(demands, axis=0)
    max_demand_per_res[max_demand_per_res == 0] = 1.0
    norm_demands = demands / max_demand_per_res
    inv_tightness = 1.0 / (resource_tightness + 1e-10)
    resource_criticality = np.sum(norm_demands * inv_tightness, axis=1)

    # Contention: sum of shared demands with successors
    contention = np.zeros(n)
    for j in range(n):
        for v in succs[j]:
            # Sum over resources of min(demand[j,r], demand[v,r])
            shared = np.minimum(demands[j], demands[v])
            contention[j] += np.sum(shared)
    
    pred_count = np.zeros(n)
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            pred_count[j] = len(p)

    succ_count = np.zeros(n)
    for j in range(n):
        succ_count[j] = len(succs[j])

    dur = durations.copy()

    feat_slack = normalize(slack)
    feat_rc = normalize(resource_criticality)
    feat_dur = normalize(dur)
    feat_pred = normalize(pred_count)
    feat_succ = normalize(succ_count)
    feat_es = normalize(ES)
    feat_contention = normalize(contention)

    # Base priority: composite of features
    # Low slack = high priority, high RC = high priority, high contention = high priority
    base_priority = (0.4 * (1.0 - feat_slack) + 
                     0.25 * feat_rc + 
                     0.2 * feat_contention + 
                     0.1 * feat_dur + 
                     0.05 * (1.0 - feat_es))

    # --- Fast SGS Simulator ---
    # Precompute predecessor sets for speed
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)

    def simulate_sgs(prio_scores, start_time_override=None, initial_scheduled=None):
        """
        Optimized Serial SGS using event-based resource tracking.
        """
        t_start = start_time if start_time_override is None else start_time_override
        if time.time() - t_start > DEADLINE:
            return float('inf')

        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        if initial_scheduled:
            for idx, (st, ft) in initial_scheduled.items():
                start_t[idx] = st
                finish_t[idx] = ft

        remaining = set(range(n))
        if initial_scheduled:
            remaining -= set(initial_scheduled.keys())

        # Track current resource usage and upcoming events
        # current_usage[r] is the usage at time 'current_time'
        # events: list of (time, delta_vector) sorted by time
        current_usage = np.zeros(R)
        events = [] # list of (time, index_in_events_to_sort, delta) - actually just keep a list and sort when needed or use heap
        # For efficiency in finding earliest feasible time, we need to know when resources free up.
        # We can maintain a sorted list of (time, delta_vector) or just check candidate times.
        # Candidate times are: current_time, and finish times of any active task.
        
        # To speed up finding the next feasible time, we can maintain a min-heap of finish times of active tasks.
        # But since we need to check feasibility at specific times, let's stick to the candidate time approach
        # but optimize the feasibility check.
        
        # Instead of checking all n tasks for each candidate time, we can maintain a list of active tasks.
        # Or, better: Pre-compute the set of "relevant" times.
        
        # Let's use a different optimization:
        # Maintain a list of (finish_time, task_id) for all scheduled tasks.
        # When checking a new task, we only need to consider finish times of tasks that overlap with [t, t+dur].
        
        # Actually, the standard optimization is:
        # 1. Maintain a sorted list of all "breakpoints" (start and finish times of scheduled tasks).
        # 2. For a new task, the earliest feasible start time is either the ES or the first breakpoint >= ES where resources are sufficient.
        
        # Let's implement a fast check:
        # We keep a list of all scheduled tasks.
        # To find the earliest feasible time for a new task:
        # 1. Start at ES.
        # 2. Check if resources are available at ES.
        # 3. If not, jump to the next finish time of any task that is currently consuming resources.
        
        # To do this efficiently, we can maintain a min-heap of finish times of all active tasks.
        # But "active" changes over time.
        
        # Alternative: Just iterate through candidate times (finish times of all scheduled tasks >= ES) in sorted order.
        # This is O(N) candidates, and each check is O(N) if naive.
        # To make check O(1) or O(log N), we can pre-compute resource usage at each breakpoint.
        
        # Let's try a simpler optimization:
        # Use a list to store scheduled tasks.
        # For each candidate time t, calculate the resource usage by summing demands of tasks that overlap [t, t+dur].
        # We can speed this up by only checking tasks that are active at t.
        
        # Let's stick to the candidate time approach but optimize the inner loop.
        
        while remaining:
            if time.time() - t_start > DEADLINE:
                return float('inf')

            # Find eligible
            eligible = []
            for j in remaining:
                is_eligible = True
                for p in pred_set[j]:
                    if start_t[p] == np.inf:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(j)

            if not eligible:
                # Fallback
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
            
            cand_times = sorted(set(cand_times))

            best_start = es_best
            found = False
            for t in cand_times:
                if t < es_best:
                    continue
                end_t = t + durations[best]
                
                # Check resource feasibility
                # Only check tasks that overlap with [t, end_t]
                feasible = True
                # We need to check resource usage at all points in [t, end_t] where usage changes.
                # Usage changes at start and finish times of overlapping tasks.
                
                # To be safe and fast, let's just check at the start time t and at every finish time of an overlapping task within (t, end_t).
                # Actually, the maximum usage occurs at the start of some task or the finish of some task?
                # No, the usage is constant between breakpoints. The breakpoints are start and finish times.
                # So we need to check at t, and at every start/finish time of any task that overlaps [t, end_t], that lies within (t, end_t).
                
                # Let's collect these points.
                check_points = {t}
                for k in range(n):
                    if start_t[k] != np.inf:
                        # Check if task k overlaps with [t, end_t]
                        if start_t[k] < end_t and finish_t[k] > t:
                            if t < start_t[k] < end_t:
                                check_points.add(start_t[k])
                            if t < finish_t[k] < end_t:
                                check_points.add(finish_t[k])
                
                # We also need to check at end_t? No, because the new task finishes at end_t, so it doesn't consume resources after end_t.
                # But we need to ensure that at any time in [t, end_t), the total usage (including new task) <= capacity.
                # The usage is constant on intervals between breakpoints.
                # So checking at the start of each interval is sufficient.
                # The intervals are [t, p1), [p1, p2), ..., [pk, end_t).
                # So we need to check at t, p1, p2, ..., pk.
                
                check_points = sorted(check_points)
                
                for i in range(len(check_points)):
                    tp = check_points[i]
                    if tp >= end_t:
                        break
                    # Calculate usage at tp
                    # Usage at tp = sum of demands of all tasks k such that start_t[k] <= tp < finish_t[k]
                    # Plus the new task's demand.
                    
                    # To speed this up, we can pre-compute the usage at all breakpoints?
                    # No, because the set of scheduled tasks changes.
                    
                    # Let's just sum it up.
                    # Optimization: Only sum tasks that are active at tp.
                    # We can maintain a list of active tasks?
                    
                    # For now, let's just iterate.
                    active_demand = np.zeros(R)
                    for k in range(n):
                        if start_t[k] != np.inf:
                            if start_t[k] <= tp and finish_t[k] > tp:
                                active_demand += demands[k]
                    
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

    # --- Local Search with Better Initialization ---
    # Try several heuristic priority rules and pick the best one as the starting point
    # for local search.
    
    # Heuristic 1: Min Slack
    h1 = 1.0 - feat_slack
    
    # Heuristic 2: Min Slack + Resource Criticality
    h2 = 0.7 * (1.0 - feat_slack) + 0.3 * feat_rc
    
    # Heuristic 3: Min Slack + Contention
    h3 = 0.7 * (1.0 - feat_slack) + 0.3 * feat_contention
    
    # Heuristic 4: Duration + Slack
    h4 = 0.5 * feat_dur + 0.5 * (1.0 - feat_slack)
    
    # Heuristic 5: The original base_priority
    h5 = base_priority
    
    # Evaluate each heuristic
    best_init_scores = h1
    best_init_ms = simulate_sgs(h1)
    
    for h in [h2, h3, h4, h5]:
        if time.time() - start_time > DEADLINE - 10.0:
            break
        ms = simulate_sgs(h)
        if ms < best_init_ms:
            best_init_ms = ms
            best_init_scores = h

    # --- Local Search ---
    best_scores = best_init_scores.copy()
    best_ms = best_init_ms
    
    current_scores = best_scores.copy()
    current_ms = best_ms
    
    T = 0.5
    T_min = 0.01
    cooling = 0.995
    
    max_iterations = 3000
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Generate neighbor
        if random.random() < 0.5:
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], neighbor_scores[i1]
        else:
            i1 = random.randint(0, n-1)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.1)
        
        neighbor_ms = simulate_sgs(neighbor_scores)
        
        if neighbor_ms is float('inf'):
            continue
            
        delta = neighbor_ms - current_ms
        if delta < 0:
            current_scores = neighbor_scores
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_scores = current_scores.copy()
        else:
            if random.random() < np.exp(-delta / T):
                current_scores = neighbor_scores
                current_ms = neighbor_ms
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_scores
# EVOLVE-BLOCK-END

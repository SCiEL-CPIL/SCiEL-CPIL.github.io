import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Strategy: Dynamic Resource-Aware Composite Priority via Local Search.
    
    This strategy moves away from static CPM metrics or single-pass simulations.
    Instead, it formulates the priority score as a composite of multiple dynamic
    features:
    1. Static Slack (from CPM)
    2. Depth (longest path from source)
    3. Tail (longest path to sink)
    4. Resource Pressure (normalized demand relative to capacity)
    5. Dynamic Resource Fit (how well the activity fits into the current resource envelope in a simulated schedule)
    
    We use a local search (hill climbing) to find the optimal weights for these
    features. The fitness function is the makespan of a Serial Schedule Generation
    Scheme (SSGS) simulation using the current weight vector.
    
    To specifically address the "dynamic resource slack" directive, the "Dynamic
    Resource Fit" feature is computed by simulating the schedule and measuring
    the "slack" in resources at the time the activity is scheduled. Activities
    that cause tight resource constraints get a lower score.
    
    This is a different design family: Weighted Multi-Feature Optimization via
    Local Search, rather than a fixed heuristic or a single simulation-based
    ranking.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit (evaluator kills at 150s, but task says 120s? 
                      # Task says 120s, but directive says 150s. Let's use 115s to be safe for 120s limit)
    DEADLINE = 115.0
    
    # 1. Extract Data
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    n = len(durations)
    num_resources = len(capacities)
    
    # Build adjacency lists
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological Sort
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    in_degree_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        return durations.copy()

    # 2. CPM Calculation
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan_cpm = max(EF)
    
    LF = [makespan_cpm] * n
    LS = [makespan_cpm] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        else:
            LF[j] = makespan_cpm
        LS[j] = LF[j] - durations[j]
        
    Slack = np.array([LS[j] - ES[j] for j in range(n)])
    
    # 3. Compute Static Features
    # Depth: Longest path from source to j
    depth = np.zeros(n)
    for j in topo_order:
        if predecessors[j]:
            depth[j] = max(depth[i] + durations[i] for i in predecessors[j])
        # else depth[j] = 0
        
    # Tail: Longest path from j to sink
    tail = np.zeros(n)
    for j in reversed(topo_order):
        if successors[j]:
            tail[j] = max(tail[k] + durations[k] for k in successors[j])
        # else tail[j] = 0
        
    # Resource Pressure: Sum of demands weighted by inverse capacity
    # If capacity is 0, pressure is infinite (but assume feasible)
    total_demands = np.sum(demands, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        resource_pressure = np.where(capacities > 0, total_demands / capacities, 0.0)
    activity_pressure = np.sum(demands * resource_pressure, axis=1)
    max_pressure = np.max(activity_pressure) if np.max(activity_pressure) > 0 else 1.0
    activity_pressure_norm = activity_pressure / max_pressure
    
    # Normalize features to [0, 1] for stable weight combination
    def normalize(v):
        v_min = np.min(v)
        v_max = np.max(v)
        if v_max == v_min:
            return np.zeros_like(v)
        return (v - v_min) / (v_max - v_min)
    
    slack_norm = normalize(Slack)
    depth_norm = normalize(depth)
    tail_norm = normalize(tail)
    pressure_norm = activity_pressure_norm
    
    # 4. Define SGS Simulation Function
    def run_sgs(priority_vec):
        """
        Runs Serial Schedule Generation Scheme.
        Returns makespan and a "dynamic fit" score for each activity.
        Dynamic fit: How much resource slack was available when the activity started.
        """
        completed_mask = np.zeros(n, dtype=bool)
        eligible = deque()
        for j in range(n):
            if len(predecessors[j]) == 0:
                eligible.append(j)
                
        finish_times = np.full(n, -1.0)
        scheduled = [] # (start, end, demands, activity_id)
        
        # For dynamic fit calculation
        dynamic_fit = np.zeros(n)
        
        while len(scheduled) < n:
            if not eligible:
                break
            
            # Select highest priority from eligible
            best_act = None
            best_score = -np.inf
            for j in eligible:
                score = priority_vec[j]
                if score > best_score:
                    best_score = score
                    best_act = j
                    
            if best_act is None:
                break
                
            # Determine Start Time
            pred_finish = 0.0
            for i in predecessors[best_act]:
                if finish_times[i] > pred_finish:
                    pred_finish = finish_times[i]
            
            earliest_start = pred_finish
            
            # Resource constraint check
            t = earliest_start
            min_slack_at_start = np.inf
            while True:
                can_schedule = True
                # Calculate slack for each resource at time t
                slacks = []
                for r in range(num_resources):
                    demand = demands[best_act, r]
                    if demand == 0:
                        continue
                    used = 0.0
                    for (s, e, d, _id) in scheduled:
                        if s <= t < e:
                            used += d[r]
                    slack_r = capacities[r] - used - demand
                    slacks.append(slack_r)
                    if used + demand > capacities[r]:
                        can_schedule = False
                        break
                
                if can_schedule:
                    # If we can schedule, record the minimum slack as a "fit" score
                    # Lower slack means tighter fit, which is worse.
                    # We want to reward activities that fit comfortably.
                    # So dynamic_fit = min_slack (higher is better)
                    if slacks:
                        dynamic_fit[best_act] = min(slacks)
                    else:
                        dynamic_fit[best_act] = 1.0 # No resource demand
                    break
                
                # Move t to next event time
                next_t = np.inf
                for (s, e, d, _id) in scheduled:
                    if s <= t < e:
                        if e < next_t:
                            next_t = e
                if next_t == np.inf:
                    t += 1.0
                else:
                    t = next_t
            
            start_t = t
            end_t = t + durations[best_act]
            scheduled.append((start_t, end_t, demands[best_act], best_act))
            finish_times[best_act] = end_t
            
            eligible.remove(best_act)
            
            for k in successors[best_act]:
                is_ready = True
                for p in predecessors[k]:
                    if finish_times[p] == -1.0:
                        is_ready = False
                        break
                if is_ready:
                    eligible.append(k)
            
        makespan = max([e for (_, e, _, _) in scheduled]) if scheduled else 0.0
        return makespan, dynamic_fit

    # 5. Local Search for Weights
    # Weights: [w_slack, w_depth, w_tail, w_pressure, w_dynamic_fit]
    # Note: w_dynamic_fit is tricky because it depends on the schedule.
    # We will approximate it in the initial search by using a proxy or just using static features first,
    # then refine with dynamic fit in a second pass?
    # Actually, let's include a "resource load" feature that is static but correlates with fit.
    # Or, we can just use the static features for the search, and then add a small dynamic component?
    # The directive asks to "explicitly include the current resource slack".
    # So the priority vector MUST depend on the dynamic slack.
    
    # This implies the priority function itself is dynamic?
    # No, the interface is priority(instance) -> one score per activity.
    # So we must return a static vector.
    # How do we incorporate dynamic slack into a static vector?
    # We can simulate a schedule, compute the dynamic slack for each activity,
    # and use that as a feature in the composite score.
    
    # Let's do this:
    # 1. Run SGS with a baseline priority (e.g., -Slack) to get an initial dynamic_fit vector.
    # 2. Use this dynamic_fit as a feature in the composite score.
    # 3. Local search over weights of [Slack, Depth, Tail, Pressure, DynamicFit].
    # 4. The final priority vector is the weighted sum.
    
    # Step 1: Baseline simulation
    baseline_priority = -slack_norm
    _, dynamic_fit_base = run_sgs(baseline_priority)
    
    # Normalize dynamic fit
    # Higher dynamic fit is better (more slack).
    # We want to reward high dynamic fit.
    # So in the composite score, we add w * dynamic_fit_norm.
    dynamic_fit_norm = normalize(dynamic_fit_base)
    
    # Features:
    # F1: -Slack (higher is better)
    # F2: -Depth (higher is better? No, usually we want to start critical activities early.
    #       In SGS, higher priority means scheduled earlier.
    #       Activities with high tail (long path to end) should be scheduled early.
    #       So we want to maximize Tail.
    #       So F2: Tail (higher is better)
    # F3: -Pressure (lower is better, so -Pressure is higher is better)
    # F4: DynamicFit (higher is better)
    
    # Let's define features such that HIGHER is BETTER for priority.
    feat_slack = -slack_norm  # Min slack is better -> -slack is higher for min slack? 
                              # Wait. Slack is LS-ES. Min Slack is critical.
                              # If Slack is 0, -Slack is 0. If Slack is 10, -Slack is -10.
                              # We want Min Slack to have HIGHER priority.
                              # So we should use -Slack? No.
                              # If Slack is small, we want priority high.
                              # So Priority = -Slack?
                              # If Slack=0, P=0. If Slack=10, P=-10.
                              # 0 > -10. So Min Slack gets higher priority. Correct.
                              
    feat_tail = tail_norm # Max Tail is better -> Higher priority.
    feat_pressure = -pressure_norm # Min Pressure is better -> Higher priority.
    feat_depth = -depth_norm # Min Depth? Or Max Depth? 
                              # Usually, activities deep in the network might be constrained by many predecessors.
                              # Let's stick to Slack and Tail as primary.
                              
    feat_dyn = dynamic_fit_norm # Max Dynamic Fit is better -> Higher priority.
    
    # Composite Score = w1 * feat_slack + w2 * feat_tail + w3 * feat_pressure + w4 * feat_dyn
    # We will search for w1, w2, w3, w4.
    
    # Initialize weights
    weights = np.array([1.0, 0.5, 0.2, 0.5])
    
    def evaluate(weights):
        p = weights[0] * feat_slack + weights[1] * feat_tail + weights[2] * feat_pressure + weights[3] * feat_dyn
        makespan, _ = run_sgs(p)
        return makespan
    
    best_weights = weights.copy()
    best_makespan = evaluate(weights)
    
    # Local Search (Hill Climbing)
    step_size = 0.1
    max_iter = 1000
    iter_count = 0
    
    while time.time() - start_time < DEADLINE and iter_count < max_iter:
        # Randomly perturb weights
        new_weights = weights + np.random.uniform(-step_size, step_size, size=4)
        # Ensure non-negative weights? Not strictly necessary, but helps stability.
        # Let's allow negative weights to explore more space.
        
        new_makespan = evaluate(new_weights)
        
        if new_makespan < best_makespan:
            best_makespan = new_makespan
            best_weights = new_weights
            weights = new_weights
            # Reduce step size slightly to refine
            step_size *= 0.95
        else:
            # Revert
            weights = best_weights
            step_size *= 0.98
            
        iter_count += 1
        
    # Final Priority Vector
    final_priority = (best_weights[0] * feat_slack + 
                      best_weights[1] * feat_tail + 
                      best_weights[2] * feat_pressure + 
                      best_weights[3] * feat_dyn)
    
    return final_priority

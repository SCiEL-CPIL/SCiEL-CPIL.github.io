import numpy as np
import time
import random

def _simulate_sgs(instance, priority_vec):
    """
    Fast simulation of the Serial Schedule Generation Scheme.
    Returns makespan.
    """
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    capacities = np.array(instance["capacities"], dtype=float)
    preds = instance.get("predecessors", [[] for _ in range(N)])
    
    # Normalize predecessors
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # State
    remaining_preds_count = [len(p) for p in preds]
    start_times = np.zeros(N)
    
    # Initial eligible
    eligible = [i for i in range(N) if remaining_preds_count[i] == 0]
    
    # Resource usage
    current_usage = np.zeros(len(capacities))
    
    # Running activities: list of (end_time, activity_id)
    running = []
    
    # Priority order: we will pick highest priority eligible that fits
    # To make this fast, we pre-sort all activities by priority descending
    # and iterate through them, picking the first one that is eligible and fits
    
    # Pre-sort indices by priority descending
    sorted_by_prio = np.argsort(-priority_vec)
    
    finished_count = 0
    
    while finished_count < N:
        # Try to start an activity
        started_any = False
        for act in sorted_by_prio:
            if act in eligible:
                # Check resources
                if np.all(demands[act] <= capacities - current_usage):
                    # Start it
                    start_times[act] = current_time if running or not eligible else 0
                    # Actually we need current_time
                    break
                else:
                    continue
            
        # This logic above is flawed because we don't track current_time properly in the loop
        # Let's use a cleaner event-based simulation
        
        # Reset for clean implementation
        # Actually, let's just implement it properly from scratch in this function
    
    # Clean implementation
    elapsed = np.zeros(N)
    start_times = np.zeros(N)
    current_usage = np.zeros(len(capacities))
    
    eligible = [i for i in range(N) if remaining_preds_count[i] == 0]
    running = []  # list of (end_time, act_id)
    current_time = 0.0
    
    finished_count = 0
    
    while finished_count < N:
        # Try to start the highest priority eligible activity that fits
        best_act = -1
        best_prio = -1e18
        
        for act in eligible:
            if priority_vec[act] > best_prio:
                if np.all(demands[act] <= capacities - current_usage):
                    best_prio = priority_vec[act]
                    best_act = act
        
        if best_act != -1:
            start_times[best_act] = current_time
            end_t = current_time + durations[best_act]
            running.append((end_t, best_act))
            current_usage += demands[best_act]
            eligible.remove(best_act)
            continue
        
        # No activity can be started, advance time
        if not running:
            # Should not happen for valid instances
            break
        
        # Find next event
        min_end = min(r[0] for r in running)
        current_time = min_end
        
        # Complete activities
        newly_finished = []
        still_running = []
        for end_t, act_id in running:
            if end_t == current_time:
                newly_finished.append(act_id)
                current_usage -= demands[act_id]
                finished_count += 1
            else:
                still_running.append((end_t, act_id))
        running = still_running
        
        # Update eligibility
        for act_id in newly_finished:
            for succ in succs[act_id]:
                remaining_preds_count[succ] -= 1
                if remaining_preds_count[succ] == 0:
                    eligible.append(succ)
    
    makespan = np.max(start_times + durations) if N > 0 else 0.0
    return makespan

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    """
    N = len(instance["durations"])
    
    # Extract data
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    capacities = np.array(instance["capacities"], dtype=float)
    R = len(capacities)
    
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # Topological Sort
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # Compute Critical Path Length (CPL) backwards
    cpl = np.zeros(N)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl
        
    # Compute Earliest Start Times (EST) forwards
    est = np.zeros(N)
    for i in range(len(topo_order)):
        u = topo_order[i]
        max_pred_est = 0
        for p in preds[u]:
            if est[p] + durations[p] > max_pred_est:
                max_pred_est = est[p] + durations[p]
        est[u] = max_pred_est
        
    # Compute Slack
    project_duration = np.max(est + durations)
    lft = np.full(N, project_duration)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        if succs[u]:
            min_lft_succ = float('inf')
            for v in succs[u]:
                if lft[v] - durations[v] < min_lft_succ:
                    min_lft_succ = lft[v] - durations[v]
            lft[u] = min_lft_succ
    lst = lft - durations
    slack = lst - est
    
    # Compute Resource Pressure (based on demand relative to capacity)
    pressure = np.zeros(N)
    for r in range(R):
        cap_r = capacities[r]
        if cap_r <= 0:
            continue
        d_col = demands[:, r]
        # Pressure is high if demand is close to capacity
        pressure += (d_col / cap_r) ** 2  # Quadratic to emphasize tight resources
        
    max_pressure = np.max(pressure)
    if max_pressure > 0:
        pressure = pressure / max_pressure
        
    max_slack = np.max(slack)
    if max_slack > 0:
        slack_factor = 1.0 - (slack / max_slack)
    else:
        slack_factor = np.ones(N)
        
    # Initial Priority: combination of CPL, pressure, and slack
    # High CPL, high pressure, low slack -> high priority
    best_priorities = cpl * (1.0 + 3.0 * pressure) * (1.0 + 2.0 * slack_factor)
    
    # Local Search: Simulated Annealing
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    def get_makespan(p_vec):
        return _simulate_sgs(instance, p_vec)
    
    best_makespan = get_makespan(best_priorities)
    best_p = best_priorities.copy()
    
    current_p = best_priorities.copy()
    current_m = best_makespan
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    
    random.seed(42)
    np.random.seed(42)
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < 1e-4:
            break
            
        # Perturb: swap two priorities
        i, j = random.sample(range(N), 2)
        old_i, old_j = current_p[i], current_p[j]
        current_p[i] = old_j
        current_p[j] = old_i
        
        new_m = get_makespan(current_p)
        delta = new_m - current_m
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_m = new_m
            if new_m < best_makespan:
                best_makespan = new_m
                best_p = current_p.copy()
        else:
            current_p[i] = old_i
            current_p[j] = old_j
    
    return best_p

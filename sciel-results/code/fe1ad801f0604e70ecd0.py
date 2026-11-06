import numpy as np
import time
import random

def priority(instance):
    """
    Strategy: 
    1. Extract instance data.
    2. Compute a strong initial static priority using a composite heuristic 
       (CPL + Resource Intensity + Successor Count).
    3. Use Simulated Annealing (SA) to optimize the priority vector.
       - Move: Swap priorities of two random activities.
       - Objective: Minimize Makespan from Serial SGS simulation.
       - SA allows accepting worse solutions to escape local minima, 
         which is superior to simple Hill Climbing for this non-convex landscape.
    4. Return the best priority vector found.
    
    This differs from the current program by using SA instead of HC, 
    and by using a more robust initialization that considers the 
    'distilled' order of a greedy dynamic schedule.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    R = len(capacities)
    
    # Handle predecessors
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
            
    # 2. Compute Static Heuristics for Initialization
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
        
    # Compute Resource Intensity (RI)
    cap_safe = np.where(capacities > 0, capacities, 1.0)
    ratios = demands / cap_safe
    ri = np.sum(ratios, axis=1)
    
    # Compute Successor Count (depth)
    succ_count = np.zeros(N, dtype=int)
    for i in range(N):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for s in succs[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        succ_count[i] = len(visited)

    # Initial Priority: Weighted combination
    # We want high priority for: High CPL, High RI, High Successor Count
    init_priority = cpl * (1.0 + ri) + 0.1 * succ_count.astype(float)

    # 3. Fast SGS Simulation
    # Optimized for speed: N=30 is small, but we need thousands of calls.
    def simulate_sgs(priorities):
        scheduled = [False] * N
        # active: list of (finish_time, idx)
        active = []
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        
        # Pre-compute predecessor lists for faster access
        # preds is already list of lists.
        
        while True:
            # Find eligible
            eligible = []
            for i in range(N):
                if not scheduled[i]:
                    is_eligible = True
                    for p in preds[i]:
                        if not scheduled[p]:
                            is_eligible = False
                            break
                    if is_eligible:
                        eligible.append(i)
                        
            if not eligible:
                # Wait for next event
                if not active:
                    break # Should not happen if valid
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                # Process completions
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active
                continue
                
            # Sort eligible by priority (descending)
            # Optimization: Use sorted() with key.
            eligible.sort(key=lambda x: priorities[x], reverse=True)
            
            # Attempt to schedule
            any_scheduled = False
            for i in eligible:
                # Check resources
                # Vectorized check
                if np.all(remaining_res >= demands[i]):
                    scheduled[i] = True
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    
            if not any_scheduled:
                # Wait
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                new_active = []
                for f, idx in active:
                    if f > time_current:
                        new_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                active = new_active
                
        return makespan

    # 4. Simulated Annealing
    # State: current_priority vector
    # Objective: minimize makespan
    
    current_priorities = init_priority.copy()
    current_score = simulate_sgs(current_priorities)
    
    best_priorities = current_priorities.copy()
    best_score = current_score
    
    # SA Parameters
    T = 10.0  # Initial temperature
    T_min = 0.1
    alpha = 0.95  # Cooling rate
    # We will run for a fixed number of steps or until time limit
    # Estimate: 1 sim ~ 0.1ms? 10ms is safer. 145s / 10ms = 14,500 steps.
    # Let's aim for as many steps as possible.
    
    random.seed(42)
    
    while time.time() < deadline:
        if T < T_min:
            # Reheat or stop? Reheating can help escape deep local minima.
            T = T_min * 5.0 
            
        # Generate neighbor: Swap two random indices
        i, j = random.sample(range(N), 2)
        
        # Swap
        temp_val = current_priorities[i]
        current_priorities[i] = current_priorities[j]
        current_priorities[j] = temp_val
        
        new_score = simulate_sgs(current_priorities)
        
        delta = new_score - current_score
        
        if delta < 0:
            # Better solution, always accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_priorities = current_priorities.copy()
        else:
            # Worse solution, accept with probability exp(-delta/T)
            # If delta is large, exp is small.
            # Avoid log(0) issues
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject: swap back
                    current_priorities[j] = current_priorities[i]
                    current_priorities[i] = temp_val
            # If delta == 0, it's the same, we can keep it or swap back. Keeping is fine.
            
        T *= alpha
        
    return best_priorities

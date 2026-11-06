"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Strategy: Dynamic Resource-Aware Critical Path + Restart-based Local Search.
    """
    start_time = time.time()
    # Safety margin: 135s out of 150s limit
    deadline = start_time + 135.0

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
                
    # Compute Static Critical Path Length (CPL) backwards
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
    ratios = demands / cap_safe  # Shape (N, R)
    ri = np.sum(ratios, axis= 1)  # Shape (N,)

    # 2. Fast SGS Simulation
    # We simulate based on a fixed priority vector to evaluate candidates.
    # To make it fast, we pre-sort activities by priority once, but since SGS
    # is dynamic, we must check eligibility and resources at each step.
    
    def simulate_sgs(priorities):
        """
        Serial SGS simulation.
        Returns makespan.
        """
        scheduled = [False] * N
        # active: list of (finish_time, idx)
        active = []
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        scheduled_count = 0
        
        # Pre-sort indices by priority descending
        # Note: In standard SGS, we pick the highest priority ELIGIBLE activity.
        # So we iterate through the sorted list and pick the first one that fits.
        sorted_indices = np.argsort(priorities)[::-1]
        
        while scheduled_count < N:
            # Find the highest priority eligible activity that fits in resources
            found = False
            for i in sorted_indices:
                if scheduled[i]:
                    continue
                # Check precedence
                is_eligible = True
                for p in preds[i]:
                    if not scheduled[p]:
                        is_eligible = False
                        break
                if not is_eligible:
                    continue
                # Check resources
                if np.all(remaining_res >= demands[i]):
                    # Schedule it
                    scheduled[i] = True
                    scheduled_count += 1
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    found = True
                    break
            
            if not found:
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
                
        return makespan

    # 3. Generate Initial Candidates
    candidates = []
    
    # Heuristic 1: CPL
    candidates.append(cpl.copy())
    
    # Heuristic 2: CPL * (1 + RI)
    candidates.append(cpl * (1.0 + ri))
    
    # Heuristic 3: Duration * (1 + 2*RI)
    candidates.append(durations * (1.0 + 2.0 * ri))
    
    # Heuristic 4: LPT (Duration)
    candidates.append(durations.copy())
    
    # Heuristic 5: Critical Path + Resource Weighted Slack (approx)
    # We don't have slack directly, but high RI + high CPL is a good proxy
    candidates.append(cpl + 5.0 * ri)

    # 4. Local Search (Restart-based Hill Climbing)
    best_prio = None
    best_ms = float('inf')
    
    # Evaluate initial candidates
    for p in candidates:
        ms = simulate_sgs(p)
        if ms < best_ms:
            best_ms = ms
            best_prio = p.copy()
            
    # Convert best_prio to a permutation (order of indices)
    # The SGS uses the order to pick activities.
    def perm_to_prio(order):
        prio = np.zeros(N)
        for rank, idx in enumerate(order):
            prio[idx] = N - rank
        return prio

    def prio_to_perm(prio):
        return np.argsort(prio)[::-1]

    current_order = prio_to_perm(best_prio)
    current_ms = best_ms
    
    # Restart-based Local Search
    # We will run multiple short hill-climbing sessions.
    # If no improvement is found in a session, we restart from the best global solution
    # with a random perturbation.
    
    max_total_iters = 10000
    iters_per_restart = 200
    restarts = 0
    total_iters = 0
    no_improve_count = 0
    
    while time.time() < deadline and total_iters < max_total_iters:
        session_start_time = time.time()
        session_no_improve = 0
        session_iters = 0
        
        # Local search session
        while session_iters < iters_per_restart and time.time() < deadline:
            # Randomly pick two indices in the order and swap them
            i, j = random.sample(range(N), 2)
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_prio = perm_to_prio(new_order)
            new_ms = simulate_sgs(new_prio)
            
            if new_ms < current_ms:
                current_ms = new_ms
                current_order = new_order
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_prio = new_prio.copy()
                session_no_improve = 0
            else:
                session_no_improve += 1
                
            session_iters += 1
            total_iters += 1
            
            # Early stop if we've had a long dry spell within this session
            if session_no_improve > 50:
                break
        
        # Restart logic
        if session_iters < iters_per_restart:
            # We broke early due to stagnation in this session
            # Perturb the best known solution to restart
            restarts += 1
            base_order = prio_to_perm(best_prio)
            # Apply a few random swaps to base_order
            for _ in range(3):
                i, j = random.sample(range(N), 2)
                base_order[i], base_order[j] = base_order[j], base_order[i]
            current_order = base_order
            current_prio = perm_to_prio(current_order)
            current_ms = simulate_sgs(current_prio)
            # If this perturbed version is worse than best, that's fine, we explore from here
            # But we keep best_ms and best_prio as the global best
            
    return best_prio

# EVOLVE-BLOCK-END

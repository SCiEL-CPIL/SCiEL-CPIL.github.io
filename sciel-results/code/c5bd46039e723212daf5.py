"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def priority(instance):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Parse predecessors
    preds = instance.get("predecessors", [])
    if not preds:
        preds = [[] for _ in range(n)]
    
    # Ensure preds are lists of indices
    parsed_preds = []
    for i in range(n):
        p = preds[i]
        if isinstance(p, (list, tuple, np.ndarray)):
            parsed_preds.append(list(p))
        elif isinstance(p, int):
            parsed_preds.append([p])
        else:
            parsed_preds.append([])

    # Build successors and indegrees
    succs = [[] for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for p in parsed_preds[i]:
            if 0 <= p < n:
                succs[p].append(i)
                indegree[i] += 1

    # Helper: Calculate Rank (Longest Tail) - standard strong baseline
    # We need a topological order or just reverse processing.
    # Since n is small (30), we can do a simple DFS or iterative relaxation.
    # Let's use a simple iterative relaxation for longest path from node to sink.
    # rank[i] = times[i] + max(rank[j] for j in succs[i])
    # This requires processing in reverse topological order.
    
    # Get reverse topological order (sinks first)
    visited = [False] * n
    topo_order = []
    
    # Use iterative DFS to avoid recursion limits, though 30 is safe.
    # We want sinks first, so we push to stack, and when a node has no unvisited successors, we add it.
    # Actually, standard DFS post-order gives reverse topological order (sinks first) if we traverse edges u->v.
    # Wait, if u->v means u precedes v, then v is a successor.
    # If we visit u, then visit v, then v is added before u?
    # Yes, post-order DFS: visit children, then add self.
    # So topo_order will have sinks (no successors) first? 
    # Let's trace: 1 -> 2. 
    # DFS(1): visit 2. DFS(2): no succ, add 2. Back to 1, add 1.
    # Order: [2, 1]. 
    # Rank[2] = t[2]. Rank[1] = t[1] + Rank[2].
    # This works.
    
    def dfs(u):
        if visited[u]:
            return
        visited[u] = True
        for v in succs[u]:
            dfs(v)
        topo_order.append(u)
        
    for i in range(n):
        if not visited[i]:
            dfs(i)
            
    # topo_order is [sink, ..., source]
    rank = [0.0] * n
    for u in topo_order:
        max_succ_rank = 0.0
        for v in succs[u]:
            if rank[v] > max_succ_rank:
                max_succ_rank = rank[v]
        rank[u] = times[u] + max_succ_rank
    rank = np.array(rank, dtype=float)
    
    # Helper: Count successors (out-degree)
    out_deg = np.array([len(s) for s in succs], dtype=float)
    
    # Helper: Greedy Pack Simulator
    # Takes a priority vector, returns number of stations used.
    def simulate(priority_vec):
        assigned = [False] * n
        stations = [] # list of lists of task indices
        station_times = [] # remaining time in current station? No, let's track used time.
        
        # We need to pick eligible tasks.
        # Eligible: all predecessors assigned.
        # To make it fast, we can just check preds each time. N=30 is small.
        
        current_station_used = 0.0
        num_stations = 1
        
        # Pre-compute pred sets for faster check? 
        # pred_sets[i] = set(parsed_preds[i])
        pred_sets = [set(p) for p in parsed_preds]
        
        # Track assigned preds count for each task? 
        # Or just check if all preds are in assigned_set.
        assigned_set = set()
        
        while len(assigned_set) < n:
            # Find eligible tasks
            eligible = []
            for i in range(n):
                if i not in assigned_set:
                    # Check if all preds are assigned
                    if all(p in assigned_set for p in pred_sets[i]):
                        # Check if it fits
                        if times[i] <= cycle_time - current_station_used:
                            eligible.append(i)
            
            if not eligible:
                # Start new station
                num_stations += 1
                current_station_used = 0.0
                # Re-find eligible (they are the same, just now fits better)
                eligible = []
                for i in range(n):
                    if i not in assigned_set:
                        if all(p in assigned_set for p in pred_sets[i]):
                            if times[i] <= cycle_time:
                                eligible.append(i)
                
                if not eligible:
                    # Infeasible? Should not happen for valid ALB instances with CT >= max time.
                    # If it does, return infinity.
                    return float('inf')

            # Pick highest priority
            # If tie, break by index or something stable? 
            # priority_vec might have floats. 
            # Use argmax.
            best_task = -1
            best_prio = -float('inf')
            for i in eligible:
                p = priority_vec[i]
                if p > best_prio:
                    best_prio = p
                    best_task = i
            
            # Assign
            assigned_set.add(best_task)
            current_station_used += times[best_task]
            
        return num_stations

    # Initial Score Vector
    # Combine Rank and Time? 
    # Rank is usually better. Let's start with Rank.
    # Or maybe Rank + small random noise to allow exploration?
    # Or start with a weighted mix.
    
    # Strategy: Simulated Annealing on the priority vector.
    # We want to minimize stations.
    # The "state" is the priority vector.
    # We perturb the vector, simulate, and accept/reject.
    
    # Since n=30, we have 30 dimensions.
    # Perturbation: Add Gaussian noise to a subset of dimensions? Or all?
    # If we change priorities significantly, the greedy order changes drastically.
    
    # Let's try a simpler approach first: 
    # Is there a known better static heuristic?
    # "Most Immediate Successors" (MIS) + Rank?
    # Let's try a linear combination: alpha * Rank + beta * Out_Deg + gamma * Time.
    
    # But the directive says "MATERIALLY DIFFERENT... different optimization or search method".
    # So I will implement a local search over the priority vector.
    
    # Initialize with Rank
    best_priority = rank.copy()
    best_score = simulate(best_priority)
    
    # If Rank is already good, can we do better?
    # Maybe "Shortest Processing Time" among eligible is better for packing?
    # But we must respect precedence.
    
    # Let's run a quick search.
    # Time budget: ~140s.
    # Simulation cost: O(N^2 * E) roughly. N=30, E~30-60. Very fast. <1ms.
    # We can do thousands of simulations.
    
    # Search Strategy:
    # 1. Start with Rank.
    # 2. Try perturbing the priority vector.
    #    Perturbation: Add noise to a random task's priority.
    #    Or, swap priorities of two tasks?
    #    Or, use a "learning" approach: 
    #       The greedy algorithm makes choices. 
    #       If a choice leads to a new station, maybe that task's priority should have been lower?
    #       Or maybe the task that *caused* the overflow (the one that didn't fit) should have been prioritized differently?
    
    # Let's try a Hill Climbing with random restarts or SA.
    # Given the small N, SA is a good choice.
    
    T_start = 1.0
    T_end = 0.01
    alpha = 0.99
    T = T_start
    
    current_priority = rank.copy()
    current_score = best_score
    
    # To make the search effective, we need to understand how priority affects the outcome.
    # Changing priority of a task that is never eligible or always fits doesn't matter.
    # Changing priority of a task that is a "bottleneck" matters.
    
    # Let's iterate.
    iterations = 0
    max_iterations = 100000 # Safe upper bound, time limit is real constraint
    
    while time.time() < deadline:
        iterations += 1
        
        # Generate neighbor
        # Pick 1-2 random tasks to perturb
        num_perturb = random.randint(1, 2)
        indices = random.sample(range(n), num_perturb)
        
        neighbor = current_priority.copy()
        for idx in indices:
            # Add Gaussian noise
            # Scale noise based on magnitude of priorities?
            # Priorities are roughly 0-40 (sum of times).
            noise = random.gauss(0, 2.0)
            neighbor[idx] += noise
        
        neighbor_score = simulate(neighbor)
        
        # Accept/Reject
        if neighbor_score < current_score:
            current_priority = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best_priority = current_priority.copy()
            T = max(T_end, T * alpha) # Restart cooling if improvement? Or just continue?
            # Standard SA: T = T * alpha
            T = max(T_end, T * alpha)
        else:
            # Accept with probability exp(-(delta)/T)
            if neighbor_score == current_score:
                # Neutral move, accept to explore
                if random.random() < 0.5:
                    current_priority = neighbor
            else:
                delta = neighbor_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_priority = neighbor
            T = max(T_end, T * alpha)
            
        # If T is very low, we might get stuck. 
        # Restart T occasionally?
        if iterations % 1000 == 0:
            T = max(T, T_start * 0.1) # Mild restart

    # Return the best priority vector found
    return best_priority

# EVOLVE-BLOCK-END

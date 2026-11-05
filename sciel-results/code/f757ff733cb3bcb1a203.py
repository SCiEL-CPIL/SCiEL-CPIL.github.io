"""Genetic Algorithm for Assembly Line Balancing (ALB)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _simulate_greedy(order, times, succ, pred_counts_init, cycle_time):
    """
    Simulate the greedy packing process for a given priority order.
    Returns the number of stations.
    """
    n = len(times)
    pred_counts = pred_counts_init.copy()
    
    # Initial available tasks
    available = [i for i in range(n) if pred_counts[i] == 0]
    
    # Rank array for quick lookup of priority (lower rank = higher priority)
    # order is a list of tasks from highest priority to lowest.
    # So order[0] has rank 0, order[1] has rank 1, etc.
    rank = [0] * n
    for idx, task in enumerate(order):
        rank[task] = idx
        
    stations = 0
    current_load = 0.0
    assigned_count = 0
    
    # Optimization: Instead of removing from list (O(N)), we can just mark as used?
    # But we need to check "eligible" (in available) and "fits".
    # For N=30, O(N^2) is trivial.
    
    while assigned_count < n:
        best_task = -1
        best_rank = n + 1
        remaining_space = cycle_time - current_load
        
        # Find best task in available
        for t in available:
            if times[t] <= remaining_space + 1e-9:
                if rank[t] < best_rank:
                    best_rank = rank[t]
                    best_task = t
                    
        if best_task != -1:
            assigned_count += 1
            current_load += times[best_task]
            
            # Remove from available
            # For small N, list remove is fine.
            available.remove(best_task)
            
            # Update successors
            for s in succ[best_task]:
                pred_counts[s] -= 1
                if pred_counts[s] == 0:
                    available.append(s)
        else:
            if current_load > 1e-9:
                stations += 1
                current_load = 0.0
            else:
                # Should not happen if graph is valid and times > 0
                break

    if current_load > 1e-9:
        stations += 1
        
    return stations

def _generate_initial_order(times, succ, pred_counts_init, n, strategy):
    """Generate an initial priority order based on structural heuristics."""
    
    # Pre-compute features if needed
    # In-degree is pred_counts_init
    # Out-degree (successors)
    out_deg = np.array([len(s) for s in succ])
    
    # Critical Path (Longest path to end)
    # Need topological sort first
    in_degree = pred_counts_init.copy()
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    # Simple topological sort
    while queue:
        queue.sort() # Deterministic
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) != n:
        # Cycle, return random
        return random.sample(range(n), n)

    # Compute dist (longest path from node to sink)
    dist = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_d = 0.0
        for v in succ[u]:
            if dist[v] > max_d:
                max_d = dist[v]
        dist[u] = times[u] + max_d
        
    # Total successors (transitive)
    total_succ = np.zeros(n, dtype=int)
    for u in reversed(top_order):
        count = 0
        for v in succ[u]:
            count += 1 + total_succ[v]
        total_succ[u] = count
        
    if strategy == 0: # LPT
        return sorted(range(n), key=lambda x: -times[x])
    elif strategy == 1: # Most Successors
        return sorted(range(n), key=lambda x: -total_succ[x])
    elif strategy == 2: # Critical Path
        return sorted(range(n), key=lambda x: -dist[x])
    elif strategy == 3: # Hybrid CP + LPT
        # Normalize and sum
        max_t = max(times) if max(times) > 0 else 1
        max_d = max(dist) if max(dist) > 0 else 1
        scores = [dist[i]/max_d + times[i]/max_t for i in range(n)]
        return sorted(range(n), key=lambda x: -scores[x])
    elif strategy == 4: # Random
        return random.sample(range(n), n)
    elif strategy == 5: # Hybrid TotalSucc + LPT
        max_ts = max(total_succ) if max(total_succ) > 0 else 1
        scores = [total_succ[i]/max_ts + times[i]/max_t for i in range(n)]
        return sorted(range(n), key=lambda x: -scores[x])
    else:
        return random.sample(range(n), n)

def _order_crossover(parent1, parent2):
    """Order Crossover (OX1)."""
    n = len(parent1)
    if n < 3:
        return parent1.copy()
    
    # Pick two cut points
    i, j = sorted(random.sample(range(n), 2))
    
    # Copy segment from parent1
    child = [-1] * n
    child[i:j+1] = parent1[i:j+1]
    segment_set = set(parent1[i:j+1])
    
    # Fill rest with parent2's order
    fill_idx = 0
    for k in range(n):
        if child[k] == -1:
            # Find next valid gene from parent2
            gene = parent2[k]
            while gene in segment_set:
                k += 1
                if k >= n:
                    k = 0
                gene = parent2[k]
            child[fill_idx] = gene
            segment_set.add(gene) # Not strictly needed for OX1 logic if we just iterate, but good for safety? No, OX1 just takes relative order.
            fill_idx += 1
            
    # Wait, the standard OX1 logic is:
    # 1. Copy slice from P1.
    # 2. Read P2 starting after j (wrap around), and place genes not in the slice into the child starting after j (wrap around).
    
    # Let's implement standard OX1 correctly
    child = [-1] * n
    child[i:j+1] = parent1[i:j+1]
    segment_set = set(parent1[i:j+1])
    
    # Start filling from position (j+1) % n
    fill_pos = (j + 1) % n
    # Iterate parent2 starting from (j+1) % n
    p2_start = (j + 1) % n
    for k in range(n):
        gene = parent2[(p2_start + k) % n]
        if gene not in segment_set:
            child[fill_pos] = gene
            segment_set.add(gene)
            fill_pos = (fill_pos + 1) % n
            if fill_pos == i: # Wrapped around to start of segment
                break
                
    return child

def _mutate(order, mutation_rate=0.2):
    """Swap or Insert mutation."""
    n = len(order)
    new_order = order.copy()
    if random.random() < mutation_rate:
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Insert
            i, j = random.sample(range(n), 2)
            item = new_order.pop(i)
            new_order.insert(j, item)
    return new_order

def priority(instance):
    """
    Uses Genetic Algorithm to find the best static priority order.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    pred_counts_init = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
            pred_counts_init[i] += 1
            
    # Initialize population with diverse heuristics
    pop_size = 60
    population = []
    # Strategies: 0=LPT, 1=MS, 2=CP, 3=Hybrid1, 4=Random, 5=Hybrid2
    strategies = [0, 1, 2, 3, 4, 5]
    
    # First few are specific heuristics
    for s in strategies:
        population.append(_generate_initial_order(times, succ, pred_counts_init, n, s))
        
    # Fill rest with random
    while len(population) < pop_size:
        population.append(random.sample(range(n), n))
        
    # Evaluate initial population
    fitness = [_simulate_greedy(ind, times, succ, pred_counts_init, cycle_time) for ind in population]
    
    best_idx = np.argmin(fitness)
    best_order = population[best_idx].copy()
    best_score = fitness[best_idx]
    
    gen = 0
    while time.time() < deadline:
        # Selection: Tournament
        new_population = []
        
        # Elitism: Copy best individual
        new_population.append(best_order.copy())
        
        while len(new_population) < pop_size:
            # Select parents
            p1_idx = random.randrange(pop_size)
            p2_idx = random.randrange(pop_size)
            
            # Crossover
            child = _order_crossover(population[p1_idx], population[p2_idx])
            
            # Mutation
            child = _mutate(child, mutation_rate=0.1)
            
            new_population.append(child)
            
        # Evaluate new population
        fitness = [_simulate_greedy(ind, times, succ, pred_counts_init, cycle_time) for ind in new_population]
        
        # Update best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_score:
            best_score = fitness[current_best_idx]
            best_order = new_population[current_best_idx].copy()
            
        # Replace population
        population = new_population
        gen += 1
        
    # Local polish on the best order found
    # Try all single insertions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                new_order = best_order.copy()
                item = new_order.pop(i)
                new_order.insert(j, item)
                new_score = _simulate_greedy(new_order, times, succ, pred_counts_init, cycle_time)
                if new_score < best_score:
                    best_score = new_score
                    best_order = new_order
                    improved = True
                    break
            if improved:
                break
                
    # Convert best_order to scores
    scores = np.zeros(n)
    for idx, task in enumerate(best_order):
        scores[task] = n - idx
        
    return scores
# EVOLVE-BLOCK-END

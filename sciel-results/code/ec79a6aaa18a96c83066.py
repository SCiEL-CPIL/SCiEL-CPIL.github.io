import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Local Search Optimization on Vertex Permutation
    1. Initialize the vertex ordering using a heuristic (e.g., Lowest Degree First + Hub Blocking).
    2. Simulate the greedy decoder to get the initial Independent Set size.
    3. Iteratively improve the ordering via pairwise swaps:
       - For each pair of adjacent vertices in the ordering, check if swapping them increases the IS size.
       - If so, perform the swap.
       - Repeat until no improvement is found or time limit is reached.
    4. Return the final ordering as scores (higher index = higher priority).
    
    This approach directly optimizes the objective (IS size) rather than relying on a proxy metric.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Initial ordering: Lowest Degree First with Hub Blocking bonus
    orig_deg = adj.sum(axis=1)
    orig_neighbor_deg_sum = adj @ orig_deg
    safe_orig_deg = orig_deg + 1.0
    hub_term = orig_neighbor_deg_sum / safe_orig_deg
    
    # Score for initial ordering: Lower is better (LDF)
    init_score = orig_deg - 0.5 * hub_term
    order = np.argsort(init_score)  # Ascending: low degree first
    
    # Greedy decoder simulation
    def simulate(order):
        active = np.ones(n, dtype=bool)
        picked = []
        for v in order:
            if active[v]:
                picked.append(v)
                # Deactivate v and its neighbors
                active[v] = False
                neighbors = np.where(adj[v])[0]
                active[neighbors] = False
        return len(picked)
    
    current_size = simulate(order)
    best_order = order.copy()
    best_size = current_size
    
    # Local search: Try swapping adjacent elements
    # To speed up, we can be smart about which swaps to try.
    # But for N=200, O(N^2) swaps * O(N) sim = O(N^3) = 8M ops.
    # Sim is O(N^2) worst case (deactivating neighbors).
    # Actually, sim is O(N+E) per call? No, it iterates N times, each O(E_v).
    # Total sim cost ~ O(N+E).
    # So O(N^2) swaps * O(N+E) sim ~ O(N^2 * (N+E)).
    # For N=200, E~2000, this is 40000 * 2200 ~ 88M ops. Might be slow in Python.
    
    # Optimization: Incremental update? Hard for arbitrary swaps.
    # Let's try a limited number of iterations or a faster heuristic for swaps.
    
    # Alternative: Random restarts or Hill Climbing with random neighbors?
    # Let's stick to deterministic hill climbing but with early termination.
    
    improved = True
    max_iters = 10  # Limit iterations to stay within time
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            
            # Swap i and i+1
            order[i], order[i+1] = order[i+1], order[i]
            new_size = simulate(order)
            
            if new_size > current_size:
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_order = order.copy()
            else:
                # Swap back
                order[i], order[i+1] = order[i+1], order[i]
                
        # If no improvement in this pass, we might be stuck in a local optimum.
        # We could try larger jumps (swap i and i+k), but let's keep it simple.
        
    # If we have time, try a few random restarts or perturbations?
    # For now, return the best found.
    
    # Convert order to scores
    # Vertices picked earlier should have higher scores.
    # order[0] is picked first -> highest score.
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
    
    return scores

# Fallback for safety
def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    orig_deg = adj.sum(axis=1)
    orig_neighbor_deg_sum = adj @ orig_deg
    safe_orig_deg = orig_deg + 1.0
    hub_term = orig_neighbor_deg_sum / safe_orig_deg
    
    init_score = orig_deg - 0.5 * hub_term
    order = np.argsort(init_score)
    
    def simulate(order):
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                neighbors = np.where(adj[v])[0]
                active[neighbors] = False
        return count
    
    current_size = simulate(order)
    best_order = order.copy()
    best_size = current_size
    
    improved = True
    max_iters = 5  # Fewer iterations to be safe
    
    while improved and time.time() < start_time + 140.0:
        improved = False
        for i in range(n - 1):
            if time.time() > start_time + 140.0:
                break
            
            order[i], order[i+1] = order[i+1], order[i]
            new_size = simulate(order)
            
            if new_size > current_size:
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_order = order.copy()
            else:
                order[i], order[i+1] = order[i+1], order[i]
                
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
    
    return scores

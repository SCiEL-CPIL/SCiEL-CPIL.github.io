"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute neighbor masks for fast blocking
    adj_bool = adj > 0.5
    neighbor_masks = [adj_bool[i] for i in range(n)]
    
    def evaluate_order(order):
        """
        Given a permutation of vertices (order), simulate the greedy MIS selection.
        Returns the size of the independent set.
        """
        selected = np.zeros(n, dtype=bool)
        count = 0
        # We iterate through the order. If a vertex is not yet selected (blocked),
        # we select it and block its neighbors.
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                # Block neighbors
                # selected[neighbor_masks[v]] = True
                # This is a boolean array assignment, which is fast in numpy
                selected[neighbor_masks[v]] = True
        return count

    # 1. Initial Heuristic: Sort by degree (ascending) or some other simple metric
    # Low degree is generally a good start for greedy MIS.
    degrees = adj.sum(axis=1)
    # Sort indices by degree (ascending)
    initial_order = np.argsort(degrees)
    
    # Also try a few other simple heuristics to pick the best starting point
    candidates = []
    candidates.append(initial_order)
    
    # Random shuffle
    random_order = np.arange(n)
    np.random.shuffle(random_order)
    candidates.append(random_order)
    
    # Descending degree (sometimes works in sparse graphs?)
    candidates.append(np.argsort(-degrees))
    
    # Weighted degree: degree * avg_neighbor_degree
    avg_neigh_deg = np.zeros(n)
    mask = degrees > 0
    if np.any(mask):
        avg_neigh_deg[mask] = (adj @ degrees)[mask] / degrees[mask]
    weighted_deg = degrees * (1 + avg_neigh_deg)
    candidates.append(np.argsort(weighted_deg))
    
    best_order = None
    best_size = -1
    
    for c in candidates:
        size = evaluate_order(c)
        if size > best_size:
            best_size = size
            best_order = c.copy()
            
    current_order = best_order.copy()
    current_size = best_size
    
    # 2. Iterated Local Search (ILS)
    # Perturbation: Reverse a random segment or swap random elements
    # Local Search: Try to improve by swapping adjacent elements or random pairs
    # Actually, for greedy MIS, the "local search" on the permutation is tricky.
    # A swap changes the order. We can just do random restarts with perturbations.
    # But pure random restarts are slow.
    # Let's try a simple hill-climbing on swaps:
    # Pick two random indices i, j in the order. Swap them. If size improves, keep.
    # This is slow if we do it one by one.
    
    # Alternative: Simulated Annealing on the permutation.
    # State: order
    # Move: Swap two random elements
    # Cost: -size (we want to maximize size)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Estimate time per evaluation.
    # evaluate_order is O(N^2) worst case if we use numpy boolean assignment?
    # Actually, `selected[neighbor_masks[v]] = True` is O(degree(v)).
    # Total work per evaluation is O(N + E). For N=200, E~1000-2000. Very fast.
    # We can do many iterations.
    
    max_steps = 50000
    steps = 0
    
    # To make it faster, we can pre-compute the "impact" of a swap? No, too complex.
    # Just run SA.
    
    # Optimization: Use a list for order to make swapping O(1)
    current_list = current_order.tolist()
    
    # We need to map from list to numpy array for evaluation? 
    # evaluate_order takes a numpy array or list. It iterates. 
    # Iterating a list in Python is fast.
    
    def evaluate_list(order_list):
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order_list:
            if not selected[v]:
                count += 1
                selected[v] = True
                selected[neighbor_masks[v]] = True
        return count

    current_size = evaluate_list(current_list)
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        # Pick two random distinct indices
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        # Swap
        current_list[i], current_list[j] = current_list[j], current_list[i]
        
        new_size = evaluate_list(current_list)
        
        delta = new_size - current_size
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_order = np.array(current_list)
        else:
            # Revert
            current_list[i], current_list[j] = current_list[j], current_list[i]
            
        T = max(T_min, T * cooling)
        
    # Final Polish: Local search around best_order
    # Try all single swaps? O(N^2) swaps. 40k swaps. Each eval O(N+E).
    # 40k * 2000 ops = 80M ops. Might take a few seconds. Worth it.
    
    best_list = best_order.tolist()
    best_size_final = best_size
    
    # We have ~5s left. Let's do a few passes of "try all swaps" if time permits.
    # Or just a few random restarts of the SA with lower T?
    
    # Let's do a deterministic local search: 
    # Iterate through all pairs (i, j). If swap improves, do it. Repeat until no improvement.
    # This is expensive. 
    # Instead, let's just run the SA for the remaining time with a lower T.
    
    T = 0.05
    cooling = 0.999
    steps2 = 0
    while time.time() < deadline - 1.0:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        current_list[i], current_list[j] = current_list[j], current_list[i]
        new_size = evaluate_list(current_list)
        
        delta = new_size - current_size
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_size = new_size
            if current_size > best_size_final:
                best_size_final = current_size
                best_list = current_list.copy()
                best_order = np.array(best_list)
        else:
            current_list[i], current_list[j] = current_list[j], current_list[i]
            
        T = max(0.001, T * cooling)
        
    # Return scores that induce this order.
    # The greedy decoder sorts by descending score.
    # We want the vertex at position 0 in best_order to have the highest score.
    # Vertex at position 1 to have the second highest, etc.
    # So score[v] = n - 1 - index_of_v_in_order
    
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - 1 - idx
        
    return scores
# EVOLVE-BLOCK-END

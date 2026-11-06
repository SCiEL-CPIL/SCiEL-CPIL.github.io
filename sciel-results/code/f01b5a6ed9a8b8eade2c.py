"""Improved program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            # Update available mask for neighbors
            # adjacency_bool is a boolean matrix
            available[adjacency_bool[v]] = False
    return count


def _evaluate_order(adjacency_bool, order):
    """Evaluate a priority vector by running the greedy decoder."""
    return _greedy_size(adjacency_bool, order)


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    # Compute neighborhood edge counts for each vertex (clustering proxy)
    # C[v] = number of edges among neighbors of v
    C = np.zeros(n)
    for v in range(n):
        if deg[v] < 2:
            C[v] = 0.0
        else:
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                sub = A[np.ix_(nbrs, nbrs)]
                C[v] = sub.sum() / 2.0

    # Heuristic 1: Ascending Degree
    order1 = np.argsort(deg, kind='stable')
    
    # Heuristic 2: Ascending Degree, then Ascending Clustering
    # We want low degree and low clustering
    # Score = degree + 0.1 * clustering
    score_h2 = deg + 0.1 * C
    order2 = np.argsort(score_h2, kind='stable')
    
    # Heuristic 3: Random
    rng = np.random.default_rng(42)
    order3 = rng.permutation(n)
    
    # Heuristic 4: Degree + Noise
    noise = rng.normal(0, 0.5, size=n)
    score_h4 = deg + noise
    order4 = np.argsort(score_h4, kind='stable')
    
    candidates = [order1, order2, order3, order4]
    
    best_size = -1
    best_order = None
    
    # Evaluate initial candidates
    for order in candidates:
        if time.time() > deadline:
            break
        size = _evaluate_order(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
            
    # Local Search: Multi-start Hill Climbing
    # We will perform several restarts. In each restart, we start from a perturbed version
    # of the current best or a random heuristic, and do hill climbing.
    
    # To ensure we don't timeout, we estimate time per evaluation.
    # N=200, greedy is fast. Let's assume 1ms per eval.
    # 110s allows ~100,000 evals.
    
    num_restarts = 20
    max_iters_per_restart = 2000 # 20 * 2000 = 40,000 evals
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize current order for this restart
        if r == 0:
            current_order = best_order.copy()
        else:
            # Perturb the best order found so far
            current_order = best_order.copy()
            # Apply some random swaps
            for _ in range(10):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
        current_size = _evaluate_order(A, current_order)
        
        # If this restart starts better than global best, update
        if current_size > best_size:
            best_size = current_size
            best_order = current_order.copy()
            
        no_improve_count = 0
        max_no_improve = 100
        
        for i in range(max_iters_per_restart):
            if time.time() > deadline - 5:
                break
                
            if no_improve_count > max_no_improve:
                break
                
            # Move: Pick a random vertex and move it to a random position
            v_idx = rng.integers(0, n)
            target_pos = rng.integers(0, n)
            
            if v_idx == target_pos:
                continue
                
            v = current_order[v_idx]
            
            # Simulate move
            # Remove v from v_idx and insert at target_pos
            # To avoid creating a new array every time, we can just swap?
            # Swapping is a weaker move. Moving is stronger.
            # Let's stick to swaps for speed, but try more of them.
            # Actually, let's try "Insertion" but implement it efficiently?
            # Creating a new list of 200 ints is cheap.
            
            new_order = current_order.copy()
            # Remove v
            new_order = np.delete(new_order, v_idx)
            # Insert at target_pos (adjust index if v_idx < target_pos)
            if v_idx < target_pos:
                insert_idx = target_pos - 1
            else:
                insert_idx = target_pos
            new_order = np.insert(new_order, insert_idx, v)
            
            new_size = _evaluate_order(A, new_order)
            
            if new_size > current_size:
                current_order = new_order
                current_size = new_size
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve_count += 1
                
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # So vertex at index 0 in order gets highest priority.
    # priority[v] = n - index(v)
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END

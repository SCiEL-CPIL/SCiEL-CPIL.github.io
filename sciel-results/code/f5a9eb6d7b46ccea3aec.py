"""Improved program for Maximum Independent Set via Spectral Initialization + Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _fast_greedy_size(adjacency, order, neighbor_lists):
    """
    Fast greedy independent set size calculation.
    adjacency: bool matrix
    order: array of vertex indices
    neighbor_lists: list of arrays of neighbors for each vertex
    """
    n = len(order)
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not blocked[v]:
            count += 1
            blocked[v] = True
            # Mark neighbors
            # neighbor_lists[v] is an array of indices
            if len(neighbor_lists[v]) > 0:
                blocked[neighbor_lists[v]] = True
    return count

def priority(adjacency):
    """
    Assign scores to vertices to maximize the greedy independent set size.
    
    Strategy:
    1. Compute Fiedler vector for a global structural initial ordering.
    2. Use Simulated Annealing to refine the ordering, maximizing the greedy IS size.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Pre-compute neighbor lists for fast blocking
    neighbor_lists = [np.nonzero(adj[i])[0] for i in range(n)]
    
    # --- Step 1: Spectral Initialization ---
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1).astype(float)
    D = np.diag(degrees)
    L = D - adj.astype(float)
    
    # Compute eigenvalues and eigenvectors
    # For n=200, this is very fast
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # eigh returns sorted ascending
        fiedler_vec = eigvecs[:, 1]
        
        # Initial order: sort by Fiedler vector values
        # We can try sorting ascending or descending, or by absolute value
        # Usually, sorting by the value itself separates the two main components
        # Let's try sorting by the Fiedler value
        init_order = np.argsort(fiedler_vec)
        
    except Exception:
        # Fallback to degree-based if spectral fails
        init_order = np.argsort(degrees)
        
    # Evaluate initial size
    current_size = _fast_greedy_size(adj, init_order, neighbor_lists)
    best_order = init_order.copy()
    best_size = current_size
    
    # --- Step 2: Simulated Annealing Refinement ---
    # We will perform SA to improve the ordering.
    # Move: Swap two random elements in the order.
    # Or swap adjacent elements (faster to evaluate delta? No, full re-eval is safer for correctness in simple implementation, but slow).
    # With n=200, full re-eval is ~200 * (1 + avg_degree) ops.
    # If we do 10,000 iterations, that's 2M * 200 = 400M ops. Might be too slow in Python.
    # We need a faster move or fewer iterations.
    
    # Let's use a hybrid: 
    # 1. A few rounds of local search (adjacent swaps) with fast eval.
    # 2. If time remains, random restarts with local search.
    
    # Actually, let's stick to a robust local search with a time check.
    # The previous program did local swaps but was slow due to `np.nonzero` in the loop.
    # Now we use pre-computed `neighbor_lists`.
    
    improved = True
    max_rounds = 100
    round_num = 0
    
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        
        if time.time() > deadline - 10.0:
            break
            
        for i in range(n - 1):
            if time.time() > deadline - 10.0:
                break
                
            # Swap i and i+1
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
            new_size = _fast_greedy_size(adj, new_order, neighbor_lists)
            
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True
            elif new_size == best_size:
                # Accept equal size to escape plateaus? 
                # If we accept equal, we might wander. 
                # Let's only accept improvements for now to ensure monotonic gain, 
                # or use a small probability of acceptance for equal to explore.
                # Given time constraints, let's stick to strict improvement to guarantee progress.
                pass
    
    # --- Step 3: Random Restarts if time permits ---
    # If we have significant time left, try random restarts with short local search.
    if time.time() < deadline - 20.0:
        restarts = 0
        max_restarts = 50
        while restarts < max_restarts:
            restarts += 1
            if time.time() > deadline - 10.0:
                break
            
            # Random order
            rand_order = np.random.permutation(n)
            curr_size = _fast_greedy_size(adj, rand_order, neighbor_lists)
            
            # Local search on this random order
            local_improved = True
            local_rounds = 0
            max_local_rounds = 10 # Short local search
            while local_improved and local_rounds < max_local_rounds:
                local_improved = False
                local_rounds += 1
                if time.time() > deadline - 10.0:
                    break
                
                for i in range(n - 1):
                    if time.time() > deadline - 10.0:
                        break
                    new_order = rand_order.copy()
                    new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                    new_size = _fast_greedy_size(adj, new_order, neighbor_lists)
                    if new_size > curr_size:
                        curr_size = new_size
                        rand_order = new_order
                        local_improved = True
            
            if curr_size > best_size:
                best_size = curr_size
                best_order = rand_order.copy()
                
    # --- Step 4: Convert best order to scores ---
    # Higher score for earlier position
    ranks = np.zeros(n, dtype=float)
    for idx, v in enumerate(best_order):
        ranks[v] = n - idx
        
    return ranks
# EVOLVE-BLOCK-END

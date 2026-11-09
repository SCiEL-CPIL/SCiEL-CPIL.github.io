"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Convert to float matrix for efficient matrix multiplication
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    adj_bool = adj > 0.5
    degrees = adj.sum(axis=1)
    
    # Precompute neighbor indices for each vertex for fast access
    neighbor_indices = [np.nonzero(adj_bool[i])[0] for i in range(n)]
    
    def greedy_size_from_order(order):
        """Compute greedy MIS size given a priority order (array of vertex indices)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_indices[v]) > 0:
                    selected[neighbor_indices[v]] = True
        return count

    def get_initial_order():
        """Generate a good initial order using low-degree heuristic."""
        # Low degree first is a strong baseline
        return np.argsort(degrees, kind='stable')

    best_order = get_initial_order()
    best_size = greedy_size_from_order(best_order)
    
    # Iterated Local Search (ILS)
    # We optimize the permutation directly.
    # Neighborhood: Swap two elements, or move an element.
    # Since n=200, O(N) evaluation is fast.
    
    current_order = best_order.copy()
    current_size = best_size
    
    # We will use a simple acceptance criterion: accept if better, or accept with probability p if worse (SA-like)
    # But ILS typically accepts worse moves to diversify, then restarts or continues from there.
    # Let's use a pure Hill Climbing with Random Restarts if stuck, or ILS with perturbation.
    
    # Strategy: 
    # 1. Local Search (Hill Climbing) on current_order.
    # 2. If no improvement for X steps, Perturb (swap k random elements).
    # 3. Repeat.
    
    max_no_improve = 100
    no_improve_count = 0
    
    while time.time() < deadline:
        # Local Search Phase
        improved = False
        # To speed up local search, we can iterate through random pairs until we find an improvement or check all.
        # With N=200, N^2/2 = 20,000 pairs. 
        # Checking one pair takes O(N) = 200 ops.
        # 20,000 * 200 = 4,000,000 ops. 
        # Python overhead makes this slow. 
        # We need a faster local search or fewer checks.
        
        # Faster Local Search: Random Sampling of Moves
        # Instead of checking all pairs, check a random subset per iteration.
        
        while no_improve_count < max_no_improve and time.time() < deadline - 5:
            # Pick a random pair to swap
            i, j = random.sample(range(n), 2)
            
            # Try swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            new_size = greedy_size_from_order(current_order)
            
            if new_size > current_size:
                current_size = new_size
                improved = True
                no_improve_count = 0
                
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
                no_improve_count += 1
        
        if time.time() >= deadline - 5:
            break
            
        # Perturbation Phase (if we stopped local search)
        # Swap k random elements to jump to a new region
        k = random.randint(2, 5)
        for _ in range(k):
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Evaluate perturbed state
        current_size = greedy_size_from_order(current_order)
        no_improve_count = 0
        
        # If perturbed state is worse than best, we might want to restart from best?
        # Standard ILS: keep current state even if worse, to explore.
        # But if we get stuck in a very bad local optimum, we might want to reset to best_order occasionally.
        # Let's add a "recovery" mechanism: if current_size is significantly worse than best_size,
        # reset current_order to best_order and perturb again.
        if current_size < best_size - 2:
            current_order = best_order.copy()
            current_size = best_size
            no_improve_count = 0
            # Perturb the best order
            for _ in range(k):
                i, j = random.sample(range(n), 2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
            current_size = greedy_size_from_order(current_order)

    # Convert best_order back to scores
    # The evaluator sorts by score descending. 
    # So vertex at index 0 in best_order (highest priority) should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores.astype(float)

# EVOLVE-BLOCK-END

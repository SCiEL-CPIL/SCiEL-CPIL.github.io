"""Guided local search penalty for TSP; Active target-tour bias via randomized k-opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _two_opt(tour, dist, max_passes=10):
    """
    Standard 2-opt improvement.
    tour: list of node indices
    dist: distance matrix
    max_passes: maximum number of full passes
    Returns: improved tour
    """
    n = len(tour)
    if n < 3:
        return tour
    
    tour = list(tour)
    # Precompute adjacency for speed? No, n=100 is small.
    # We will use the provided dist matrix.
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                
                # Check if edges (a,b) and (c,d) are adjacent in the tour
                if i == 0 and j == n - 1:
                    continue
                    
                # Calculate delta
                # Current: (a,b) + (c,d)
                # New: (a,c) + (b,d) OR (a,d) + (b,c)
                # Standard 2-opt reverses the segment between b and c.
                # So we connect a-c and b-d.
                
                d_ab = dist[a, b]
                d_cd = dist[c, d]
                d_ac = dist[a, c]
                d_bd = dist[b, d]
                
                if d_ac + d_bd < d_ab + d_cd:
                    # Reverse segment from i+1 to j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    # Restart inner loop? Or continue? 
                    # For efficiency in this helper, we can just continue the loop, 
                    # but restarting the outer loop is safer for correctness of "passes".
                    # However, given the small n and max_passes, simple continuation is ok.
                    # To be robust, let's break and restart the pass? 
                    # No, standard 2-opt often continues. We'll stick to simple update.
                    
    return tour

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Generates a target tour by applying a random k-opt perturbation to the current tour.
    Then, it constructs a modified distance matrix where the edges of this target tour
    are significantly reduced in cost, encouraging the evaluator's 2-opt to converge
    towards this new region.
    """
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    n = len(local_opt_tour)
    if n < 3:
        return edge_distance.copy()
    
    # 1. Generate a candidate "target" tour
    # We use a random 3-opt or 4-opt style perturbation to jump out of the local optimum.
    # We pick 3 random edges to remove and reconnect optimally (or randomly).
    # For simplicity and speed, we do a random reversal of a random segment (2-opt move)
    # but applied to a random segment, not just the best one. This is a "random 2-opt move".
    # To be more aggressive, we can do multiple random 2-opt moves.
    
    target_tour = list(local_opt_tour)
    
    # Perform a few random 2-opt moves to create a diverse target
    num_random_moves = 3
    for _ in range(num_random_moves):
        if time.time() > deadline:
            break
        i = random.randint(0, n - 2)
        j = random.randint(i + 2, n - 1)
        if i == 0 and j == n - 1:
            continue
        
        # Apply the move
        target_tour[i+1:j+1] = reversed(target_tour[i+1:j+1])
        
    # 2. Construct the biased distance matrix
    # We want the edges in target_tour to be cheap.
    # We also want to keep the general geometry somewhat intact so 2-opt doesn't fail completely.
    # Strategy: 
    # - Start with true distances.
    # - For edges in the target tour, multiply by a factor < 1.0 (e.g., 0.5) to make them attractive.
    # - For edges NOT in the target tour, we can keep them or slightly penalize them if they were in the current tour?
    #   The prompt says "add the current tour's edges to edge_n_used". 
    #   We can use edge_n_used to penalize edges that are heavily used but NOT in the target.
    
    new_dist = edge_distance.copy()
    
    # Identify edges in the target tour
    target_edges = set()
    for i in range(n):
        u = target_tour[i]
        v = target_tour[(i + 1) % n]
        target_edges.add((u, v))
        target_edges.add((v, u))
    
    # Identify edges in the current tour (to potentially penalize if not in target)
    current_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        current_edges.add((u, v))
        current_edges.add((v, u))
    
    # Apply biases
    # 1. Attract target edges
    #    Set distance to 0.5 * true_dist. This makes them very attractive.
    # 2. Penalize current edges that are NOT in the target
    #    If an edge is in current tour but not in target, and has high usage, penalize it.
    #    This helps "break" the current local optimum structure.
    
    counts = edge_n_used.astype(float)
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in target_edges:
                # Strong attraction
                new_dist[i, j] = 0.5 * edge_distance[i, j]
                new_dist[j, i] = new_dist[i, j]
            else:
                # Check if it was in the current tour
                if (i, j) in current_edges:
                    # If it's not in the target, it's "obsolete" for this jump.
                    # Penalize based on usage count to push 2-opt away from it.
                    # Factor: 1 + 0.2 * count
                    factor = 1.0 + 0.2 * counts[i, j]
                    new_dist[i, j] = edge_distance[i, j] * factor
                    new_dist[j, i] = new_dist[i, j]
                else:
                    # Leave as is, or slight penalty if used before?
                    # Let's leave as true distance to preserve geometry for non-tour edges.
                    pass

    # Ensure symmetry
    new_dist = (new_dist + new_dist.T) / 2.0
    
    # Safety: ensure finite
    if not np.all(np.isfinite(new_dist)):
        return edge_distance.copy()
        
    return new_dist
# EVOLVE-BLOCK-END

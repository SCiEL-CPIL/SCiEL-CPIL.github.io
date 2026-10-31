import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Generate candidate vectors with different weights
    # Weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2 (since norm^2 = 4, distance^2 = 8 - 2*dot, need distance >= 2 => dot <= 2)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists for fast access
    neighbor_lists = []
    for i in range(N4):
        neighbor_lists.append(np.where(conflicts[i])[0])
    
    def greedy_mis(seed, time_limit=None):
        """Greedy maximum independent set with random restart."""
        if time_limit is None:
            time_limit = deadline - 2
        rng = np.random.default_rng(seed)
        alive = np.ones(N4, dtype=bool)
        selected_indices = []
        degrees = conflicts.sum(axis=1).astype(np.int64)
        
        while np.any(alive) and time.time() < time_limit:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Randomized scoring: degree + random noise
            alive_degrees = degrees[alive_indices].astype(np.float64)
            noise = rng.random(len(alive_indices)) * 0.5
            scores = alive_degrees + noise
            
            # Select the node with minimum score
            min_score_idx = np.argmin(scores)
            chosen = alive_indices[min_score_idx]
            
            selected_indices.append(chosen)
            
            # Collect all nodes to remove: chosen + its alive neighbors
            neighbors = neighbor_lists[chosen]
            to_remove = np.append([chosen], neighbors[alive[neighbors]])
            alive[to_remove] = False
            
            # Update degrees
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                removed_neighbor_counts = conflicts[remaining][:, to_remove].sum(axis=1)
                degrees[remaining] -= removed_neighbor_counts
        
        return selected_indices
    
    # Try multiple random restarts and keep the best result
    best_selected = None
    best_size = 0
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        selected = greedy_mis(restart, deadline - 10)
        size = len(selected)
        
        if size > best_size:
            best_size = size
            best_selected = selected
        
        # If we've found a very good solution, we can stop early
        if size >= 1000:
            break
    
    if best_selected is None:
        best_selected = []
    
    result_v4 = vectors_v4[best_selected]
    
    # V1 vectors: all 22 are mutually compatible (as analyzed)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

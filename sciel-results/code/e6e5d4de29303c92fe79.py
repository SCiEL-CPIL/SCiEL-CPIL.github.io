import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
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
    
    def local_search_polish(selected_mask, max_iters=1000):
        """
        Local search: try to improve the independent set by swapping.
        For each non-selected vertex v, if we add v, we must remove all selected neighbors of v.
        Net gain = 1 - (number of selected neighbors of v).
        If net gain > 0, do the swap.
        """
        selected = selected_mask.copy()
        selected_set = set(np.where(selected)[0].tolist())
        non_selected_set = set(range(N4)) - selected_set
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # For each non-selected vertex, compute net gain
            # net_gain(v) = 1 - |selected_neighbors(v)|
            best_v = -1
            best_gain = 0
            
            for v in non_selected_set:
                # Count how many of v's neighbors are selected
                neighbors = neighbor_lists[v]
                selected_neighbors = sum(1 for n in neighbors if n in selected_set)
                gain = 1 - selected_neighbors
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v >= 0 and best_gain > 0:
                # Add best_v, remove all its selected neighbors
                neighbors = neighbor_lists[best_v]
                to_remove = [n for n in neighbors if n in selected_set]
                for n in to_remove:
                    selected[n] = False
                    selected_set.remove(n)
                    non_selected_set.add(n)
                selected[best_v] = True
                selected_set.add(best_v)
                non_selected_set.remove(best_v)
                improved = True
        
        return selected
    
    def greedy_mis(seed, time_limit=None):
        """Greedy maximum independent set with random restart."""
        if time_limit is None:
            time_limit = deadline - 2
        rng = np.random.default_rng(seed)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
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
            
            selected[chosen] = True
            
            # Collect all nodes to remove: chosen + its alive neighbors
            neighbors = neighbor_lists[chosen]
            to_remove = np.append([chosen], neighbors[alive[neighbors]])
            alive[to_remove] = False
            
            # Update degrees
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                removed_neighbor_counts = conflicts[remaining][:, to_remove].sum(axis=1)
                degrees[remaining] -= removed_neighbor_counts
        
        return selected
    
    # Try multiple random restarts and keep the best result
    best_selected = None
    best_size = 0
    num_restarts = 30  # Reduced restarts to allow time for local search
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        selected = greedy_mis(restart, deadline - 20)
        
        # Apply local search polish
        polished = local_search_polish(selected)
        size = polished.sum()
        
        if size > best_size:
            best_size = size
            best_selected = polished
        
        # If we've found a very good solution, we can stop early
        if size >= 1100:
            break
    
    if best_selected is None:
        # Fallback: just do one greedy without polish
        best_selected = greedy_mis(0, deadline - 5)
    
    result_v4 = vectors_v4[best_selected]
    
    # V1 vectors: all 22 are mutually compatible
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

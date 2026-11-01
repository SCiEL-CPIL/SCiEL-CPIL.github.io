"""Rewritten program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    # Using matrix multiplication for speed
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill(in_set, counts):
        """
        Add all possible nodes to in_set, updating counts.
        Returns number of nodes added.
        """
        added = 0
        # Nodes with count 0 and not in set can be added
        # We iterate until no more can be added
        # Since adding a node increases counts of neighbors, it might block others,
        # but never unblocks them. So a single pass over candidates is enough?
        # No, if we add A, it blocks B. If B was a candidate, it's now blocked.
        # But if C was blocked by B, and B is not added, C remains blocked.
        # So we just need to check all nodes with count 0.
        
        # Efficient way:
        # Find all v where counts[v] == 0 and not in_set[v]
        candidates = np.where((counts == 0) & (~in_set))[0]
        
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def local_search(in_set):
        """
        Simulated Annealing based local search.
        """
        in_set = in_set.copy()
        size = in_set.sum()
        
        # Ensure valid start, calculate conflicts
        global_conflicts = 0
        # If in_set is from greedy, it's valid.
        
        best_in_set = in_set.copy()
        best_size = size
        
        T = 1.0
        cooling = 0.9995
        lambda_pen = 10.0
        rng = np.random.default_rng(42)
        
        while time.time() < deadline - 10 and T > 0.01:
            v = rng.integers(N4)
            nbrs = neighbors[v]
            conflict_count = np.count_nonzero(in_set[nbrs])
            
            if in_set[v]:
                # Remove v
                delta_E = 1.0 - lambda_pen * conflict_count
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    in_set[v] = False
                    size -= 1
                    global_conflicts -= conflict_count
            else:
                # Add v
                delta_E = -1.0 + lambda_pen * conflict_count
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    in_set[v] = True
                    size += 1
                    global_conflicts += conflict_count
            
            if global_conflicts == 0 and size > best_size:
                best_size = size
                best_in_set = in_set.copy()
                
            T *= cooling
            
        # Final Polish: Greedy fill
        counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            counts[neighbors[i]] += 1
            
        candidates = np.where((counts == 0) & (~best_in_set))[0]
        for v in candidates:
            if counts[v] == 0 and not best_in_set[v]:
                best_in_set[v] = True
                for u in neighbors[v]:
                    counts[u] += 1
        best_size = best_in_set.sum()

        return best_in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        # Allocate time to local search
        # Local search is the main driver
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Optional: Try a random perturbation of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            # Remove 2 random elements
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1) if they don't conflict?
    # The prompt's seed program added them.
    # W1 vectors: (2,0,0...), etc.
    # Dot product with W4 vectors:
    # W4 has one 1 at pos i. W1 has 2 at pos i. Dot = 2.
    # Conflict if dot > 2. So Dot=2 is NOT a conflict.
    # Dot product with other W1 vectors:
    # If same pos: 4. Conflict.
    # If diff pos: 0. No conflict.
    # So we can add W1 vectors at positions not "saturated" by W4 vectors?
    # Actually, if a W4 vector has a 1 at pos i, it has dot 2 with W1_i.
    # If W4 vector has -1 at pos i, dot is -2.
    # So W1_i is compatible with all W4 vectors?
    # Max dot is 2. Condition is dot <= 2. So yes.
    # However, W1_i and W1_j (i!=j) have dot 0. Compatible.
    # W1_i and W1_i (same) not allowed (distinct vectors).
    # So we can add all 22 W1 vectors?
    # Wait, W1 vectors are (2,0..), (-2,0..).
    # v1 = (2,0..), v2 = (-2,0..).
    # Distance: 4. Norm: 2. Dist >= Norm. OK.
    # So we can add all 22 W1 vectors.
    
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END

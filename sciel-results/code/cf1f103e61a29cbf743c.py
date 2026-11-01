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
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute conflict matrix as int for fast counting
    conflicts_int = conflicts.astype(np.int32)
    
    # Convert to neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        """Greedy independent set: pick min-degree vertex, remove it and its neighbors."""
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
                degrees[rem] -= conflicts_int[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill(in_set, counts):
        """
        Add all possible nodes to in_set where counts[v]==0 and not in_set[v].
        """
        added = 0
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
        Vectorized Simulated Annealing local search.
        """
        in_set = in_set.copy()
        size = int(in_set.sum())
        
        # Track conflict count per vertex: number of in_set neighbors
        cc = conflicts_int @ in_set.astype(np.int32)
        
        best_in_set = in_set.copy()
        best_size = size
        
        T = 1.0
        cooling = 0.999
        lambda_pen = 5.0
        rng_ls = np.random.default_rng(42)
        
        while time.time() < deadline - 10 and T > 0.01:
            v = int(rng_ls.integers(N4))
            conflict_count = int(cc[v])
            
            if in_set[v]:
                # Remove v
                delta_E = 1.0 - lambda_pen * conflict_count
                if delta_E < 0 or rng_ls.random() < np.exp(-delta_E / T):
                    in_set[v] = False
                    size -= 1
                    # Update cc for neighbors of v
                    nbrs = neighbors[v]
                    if len(nbrs) > 0:
                        cc[nbrs] -= 1
                    cc[v] = 0  # no longer in set
            else:
                # Add v
                delta_E = -1.0 + lambda_pen * conflict_count
                if delta_E < 0 or rng_ls.random() < np.exp(-delta_E / T):
                    in_set[v] = True
                    size += 1
                    # Update cc for neighbors of v
                    nbrs = neighbors[v]
                    if len(nbrs) > 0:
                        cc[nbrs] += 1
                    cc[v] = 0  # v itself has no self-conflict
            
            if size > best_size:
                # Check if valid (no conflicts)
                # Quick check: sum of cc over in_set should be 0 if valid
                if int(cc[in_set].sum()) == 0:
                    best_size = size
                    best_in_set = in_set.copy()
                
            T *= cooling
            
        # Final Polish: Greedy fill on best_in_set
        counts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(best_in_set)[0]
        if len(in_set_indices) > 0:
            for i in in_set_indices:
                counts[neighbors[i]] += 1
            
        candidates = np.where((counts == 0) & (~best_in_set))[0]
        for v in candidates:
            if counts[v] == 0 and not best_in_set[v]:
                best_in_set[v] = True
                for u in neighbors[v]:
                    counts[u] += 1
        best_size = int(best_in_set.sum())

        return best_in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Perturbation of best set
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = int(perturbed.sum())
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1): (2,0,..0) and (-2,0,..0) for each position
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Add W2 vectors (weight 2, entries in {-1,0,1}, norm^2=2)
    # These have norm sqrt(2) < 2, so they need careful handling
    # Actually, the condition is: min pairwise distance >= max norm
    # If we mix vectors of different norms, we need to be careful
    # Let's check: if we have vectors with norms sqrt(4)=2 and sqrt(2), 
    # max norm = 2, so min distance must be >= 2
    # Distance between v4 (norm 2) and v2 (norm sqrt(2)):
    # ||v4 - v2||^2 = 4 + 2 - 2*v4.v2 = 6 - 2*v4.v2
    # Need >= 4, so v4.v2 <= 1
    # v4 has 4 nonzero entries of +/-1, v2 has 2 nonzero entries of +/-1
    # Max dot = 2 (if both nonzero positions overlap with same sign)
    # So v4.v2 can be 2, which gives distance^2 = 2 < 4. NOT OK.
    # So W2 vectors may conflict with W4 vectors.
    # Let's skip W2 for safety, or add them only if compatible.
    
    # For now, just add W1 vectors which are always compatible with W4
    # W1 dot W4: max is 2 (when W1 has 2 at position where W4 has 1)
    # Distance^2 = 4 + 4 - 2*2 = 4. Need >= max_norm^2 = 4. OK.
    # W1 dot W1: same pos = 4 or -4, diff pos = 0
    # Same pos: distance^2 = 4+4-2*4 = -4? No, (2,-2): dist^2 = 16. OK.
    # (2,2) same vector not allowed. (2,-2): dist^2=16 >= 4. OK.
    # (2,0) and (0,2): dist^2 = 8 >= 4. OK.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END

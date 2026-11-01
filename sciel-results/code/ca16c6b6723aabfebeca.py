import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Generate all candidate vectors with norm^2 = 4
    # Type 1: Weight 4 (four +/-1)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Type 2: Weight 1 (one +/-2)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    all_vectors = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_vectors)
    
    # 2. Compute conflict matrix
    # Conflict if dot > 2
    dots = all_vectors @ all_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    best_set = np.zeros(N, dtype=bool)
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    def greedy_init():
        in_set = np.zeros(N, dtype=bool)
        counts = np.zeros(N, dtype=np.int32)
        indices = rng.permutation(N)
        for v in indices:
            if counts[v] == 0:
                in_set[v] = True
                counts[neighbors[v]] += 1
        return in_set, counts

    def local_search_sa():
        in_set, counts = greedy_init()
        current_size = in_set.sum()
        best_local_size = current_size
        best_local_set = in_set.copy()
        
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        
        while T > min_T and time.time() < deadline - 5:
            move_type = rng.integers(3)
            
            if move_type == 0:
                # Try to add
                candidates = np.where(~in_set)[0]
                if len(candidates) > 0:
                    v = candidates[rng.integers(len(candidates))]
                    if counts[v] == 0:
                        in_set[v] = True
                        counts[neighbors[v]] += 1
                        current_size += 1
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local_set = in_set.copy()
            elif move_type == 1:
                # Try to swap (remove 1, add 1)
                in_set_indices = np.where(in_set)[0]
                not_in_set_indices = np.where(~in_set)[0]
                if len(in_set_indices) > 0 and len(not_in_set_indices) > 0:
                    v_in = in_set_indices[rng.integers(len(in_set_indices))]
                    v_out = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                    
                    # Save state
                    old_in_set = in_set.copy()
                    old_counts = counts.copy()
                    
                    # Remove v_in
                    in_set[v_in] = False
                    for u in neighbors[v_in]:
                        counts[u] -= 1
                        
                    # Try to add v_out
                    if counts[v_out] == 0:
                        in_set[v_out] = True
                        for u in neighbors[v_out]:
                            counts[u] += 1
                    else:
                        # Revert
                        in_set = old_in_set
                        counts = old_counts
            else:
                # Try to remove 1, add up to 3
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    v_in = in_set_indices[rng.integers(len(in_set_indices))]
                    
                    # Save state
                    old_in_set = in_set.copy()
                    old_counts = counts.copy()
                    old_size = current_size
                    
                    # Remove v_in
                    in_set[v_in] = False
                    for u in neighbors[v_in]:
                        counts[u] -= 1
                    current_size -= 1
                    
                    # Try to add up to 3 random vectors
                    not_in_set_indices = np.where(~in_set)[0]
                    if len(not_in_set_indices) > 0:
                        picks = not_in_set_indices[rng.permutation(len(not_in_set_indices))[:min(10, len(not_in_set_indices))]]
                        for v_out in picks:
                            if counts[v_out] == 0:
                                in_set[v_out] = True
                                for u in neighbors[v_out]:
                                    counts[u] += 1
                                current_size += 1
                                if current_size > old_size:
                                    break
                            
                    # Acceptance
                    if current_size > old_size:
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local_set = in_set.copy()
                    elif current_size < old_size:
                        # Revert
                        in_set = old_in_set
                        counts = old_counts
                        current_size = old_size
                        
            T *= cooling
            
        return best_local_set

    # Main Loop
    n_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = local_search_sa()
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()
            
        if best_size >= 100:
            break

    # Final result
    return all_vectors[best_set]

if __name__ == "__main__":
    # Test run
    start = time.time()
    res = construct()
    print(f"Size: {len(res)}, Time: {time.time() - start:.2f}s")

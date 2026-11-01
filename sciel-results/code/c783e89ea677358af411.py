"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Generate Candidates
    # Type A: W1 (Norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Type B: W4 (Norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors = np.array(v1_candidates + v4_candidates, dtype=np.int64)
    N = len(vectors)
    
    # 2. Compute Conflict Matrix
    # Conflict if dot > 2
    dots = vectors @ vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # 3. Greedy MIS with Restarts
    rng = np.random.default_rng(42)
    best_selected = None
    best_count = 0
    
    # Estimate time for one greedy run
    # N=5302.
    # We will run as many as possible.
    
    max_restarts = 100 # Will be limited by time
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Greedy Min-Degree
        # active: boolean mask
        active = np.ones(N, dtype=bool)
        selected_indices = []
        
        # To speed up degree calculation, we can maintain a degree array.
        # But updating it is tricky with vectorized ops.
        # Given N ~ 5000, recomputing degrees is O(N^2) per step? No.
        # degrees = conflicts[active].sum(axis=1) is O(N * active_count).
        # Total O(N^3) if done naively? No, sum is O(N) per node, so O(N^2) per step.
        # Total O(N^3) = 1.5 * 10^11. Too slow.
        
        # We need a faster way.
        # Use a list of active indices.
        active_list = list(range(N))
        # We need to quickly find min degree.
        # We can maintain a heap?
        # Or just sample?
        
        # Let's try a simpler heuristic: Random Greedy with "Lookahead" or just Random.
        # Random Greedy: Pick random active node, remove it and neighbors.
        # This is O(N^2) total if we use efficient removal.
        
        # Let's use a NumPy-based removal.
        active_mask = np.ones(N, dtype=bool)
        selected = []
        
        # Optimization: Pre-compute neighbor lists as sparse?
        # conflicts is dense bool.
        
        while np.any(active_mask):
            if time.time() > deadline - 5:
                break
                
            # Get active indices
            active_indices = np.where(active_mask)[0]
            if len(active_indices) == 0:
                break
            
            # Calculate degrees for active nodes
            # This is the bottleneck.
            # deg = conflicts[active_indices].sum(axis=1)
            # This creates a submatrix of size |active| x N.
            # Summing it is O(|active| * N).
            
            # To speed up, we can just pick a random node from active.
            # Random Greedy is often good enough for these sizes.
            # Let's try Random Greedy first.
            
            idx = active_indices[rng.integers(len(active_indices))]
            selected.append(idx)
            
            # Remove idx and its neighbors
            # neighbors = conflicts[idx]
            # to_remove = np.where(conflicts[idx] & active_mask)[0]
            # to_remove = np.concatenate(([idx], to_remove))
            # active_mask[to_remove] = False
            
            # Faster:
            active_mask[idx] = False
            active_mask[conflicts[idx]] = False
            
        count = len(selected)
        
        if count > best_count:
            best_count = count
            best_selected = np.zeros(N, dtype=bool)
            best_selected[selected] = True
            
    # 4. Polish with SA on the best result
    if best_selected is None:
        return vectors[:0] # Should not happen
        
    in_set = best_selected.copy()
    current_size = best_count
    best_size = current_size
    best_in_set = in_set.copy()
    
    # SA Parameters
    T = 2.0
    cooling = 0.999
    min_T = 0.1
    steps_limit = 200000
    
    for step in range(steps_limit):
        if time.time() > deadline - 2:
            break
            
        # Pick random node not in set
        not_in_set = np.where(~in_set)[0]
        if len(not_in_set) == 0:
            break
            
        v = not_in_set[rng.integers(len(not_in_set))]
        
        # Check conflicts
        # neighbors_in_set = np.where(conflicts[v] & in_set)[0]
        # k = len(neighbors_in_set)
        
        # Vectorized check:
        # mask = conflicts[v] & in_set
        # k = np.sum(mask)
        
        mask = conflicts[v] & in_set
        k = np.sum(mask)
        
        delta = 1 - k
        
        if delta > 0 or (delta == 0 and rng.random() < 0.5) or (delta < 0 and rng.random() < np.exp(delta / T)):
            # Accept
            in_set[v] = True
            if k > 0:
                # Remove neighbors
                # indices = np.where(mask)[0]
                # in_set[indices] = False
                in_set[mask] = False
            current_size += delta
            
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()
                
        T *= cooling
        if T < min_T:
            T = min_T
            
    final_indices = np.where(best_in_set)[0]
    return vectors[final_indices]

# EVOLVE-BLOCK-END

"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    # Generate all integer vectors in R^11 with squared norm 4
    # Types:
    # 1. One entry is +/-2, others 0. (2 * 11 = 22 vectors)
    # 2. Two entries are +/-1, others 0. (C(11,2) * 4 = 55 * 4 = 220 vectors)
    # 3. Four entries are +/-1, others 0. (C(11,4) * 16 = 330 * 16 = 5280 vectors)
    
    vectors_list = []
    
    # Type 1: Weight 1, value 2
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            vectors_list.append(v)
            
    # Type 2: Weight 2, value 1
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[pos1] = s1
                    v[pos2] = s2
                    vectors_list.append(v)
                    
    # Type 3: Weight 4, value 1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vectors_list.append(v)
            
    V = np.array(vectors_list, dtype=np.int64)
    N = len(V)
    
    if N == 0:
        return np.array([], dtype=np.int64).reshape(0, D)

    # Compute dot products
    # V is (N, D). Dots is (N, N)
    dots = V @ V.T
    
    # Condition: ||v_i - v_j|| >= max_norm.
    # All norms are 2 (squared 4).
    # ||v_i - v_j||^2 = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # We need 8 - 2*v_i.v_j >= 4 => 2*v_i.v_j <= 4 => v_i.v_j <= 2.
    # Conflict if v_i.v_j > 2.
    
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute adjacency lists as lists of indices
    adj = [np.where(conflicts[i])[0] for i in range(N)]
    
    # Local Search with Simulated Annealing
    # We want to maximize the size of the independent set.
    
    def local_search_sa(in_set, max_time=10.0, T_start=1.0, T_end=0.01, cooling=0.999):
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # Precompute conflict counts for each node relative to current set
        # conflict_count[i] = number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[adj[i]] += 1
            
        ls_deadline = time.time() + max_time
        T = T_start
        
        # Random node order for perturbation or just random selection
        rng = np.random.default_rng(42)
        
        while time.time() < ls_deadline:
            # Try to add a node not in the set
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            # Choose a candidate node
            # Strategy: Pick a node with low conflict count to maximize gain
            # Or pick randomly to escape local minima
            if rng.random() < 0.1:
                v = not_in[rng.integers(len(not_in))]
            else:
                cc = conflict_count[not_in]
                min_cc = np.min(cc)
                cands = not_in[cc == min_cc]
                v = cands[rng.integers(len(cands))]
                
            k = conflict_count[v]
            delta = 1 - k  # Net change in set size
            
            # Metropolis criterion
            if delta >= 0 or rng.random() < np.exp(delta / T):
                # Accept
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                # Remove conflicting neighbors
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    # Update conflict counts for neighbors of removed nodes
                    for u in to_remove:
                        conflict_count[adj[u]] -= 1
                        
                # Add v
                in_set[v] = True
                conflict_count[adj[v]] += 1
                
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                
            T *= cooling
            if T < T_end:
                T = T_end
                
        return best_in_set, best_size

    def greedy_init(seed):
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        
        # Randomized greedy: pick random node from alive, add it, remove neighbors
        # Or pick min degree?
        # Let's try random permutation greedy
        order = rng.permutation(N)
        for i in order:
            if alive[i]:
                in_set[i] = True
                # Remove i and its neighbors from alive
                to_remove = np.unique(np.concatenate(([i], adj[i])))
                alive[to_remove] = False
                
        return in_set

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Run multiple restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Get initial solution
        in_set = greedy_init(n_restarts)
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # Local Search
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
            
        ls_time = min(remaining * 0.6, 10.0)
        if ls_time < 1.0:
            ls_time = 1.0
            
        ls_result, ls_size = local_search_sa(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
        # If we made progress, run another short polish
        if ls_size > current_size and time.time() < deadline - 3:
            polish_time = min(5.0, (deadline - 3 - time.time()) * 0.5)
            if polish_time > 1.0:
                p_result, p_size = local_search_sa(ls_result, max_time=polish_time)
                if p_size > best_size:
                    best_size = p_size
                    best_in_set = p_result.copy()

    final_vectors = V[best_in_set]
    return final_vectors

# EVOLVE-BLOCK-END

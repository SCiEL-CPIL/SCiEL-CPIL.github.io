import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph using bitmask popcount
    support_masks = np.zeros(N_S, dtype=np.int32)
    for s_idx, s in enumerate(supports):
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks[s_idx] = mask

    # Efficient conflict computation: two 4-sets conflict if |intersection| >= 3
    # For 11-bit masks, we can use the fact that |A ∩ B| = popcount(A & B)
    # Precompute popcount for all 2^11 = 2048 possible masks
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(1, 2048):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Compute conflict matrix in chunks to avoid memory issues
    conflict = np.zeros((N_S, N_S), dtype=bool)
    chunk_size = 64
    for i_start in range(0, N_S, chunk_size):
        i_end = min(i_start + chunk_size, N_S)
        chunk = support_masks[i_start:i_end]
        # For each mask in chunk, compute intersection with all masks
        # Use broadcasting: chunk[:, None] & support_masks[None, :]
        inter = chunk[:, None] & support_masks[None, :]
        # inter is (chunk_size, N_S), values are 11-bit masks
        # Need popcount of each element
        # Reshape to use vectorized popcount
        inter_flat = inter.ravel()
        pc_flat = popcount[inter_flat]
        pc = pc_flat.reshape(i_end - i_start, N_S)
        conflict[i_start:i_end] = (pc >= 3)

    # Make symmetric (should already be, but ensure)
    conflict = conflict | conflict.T

    # Build adjacency lists for conflict graph
    # conflict_adj[s] is a numpy array of indices c such that conflict[s, c] is True
    conflict_adj = []
    for s in range(N_S):
        adj = np.where(conflict[s])[0]
        conflict_adj.append(adj)
        
    # Precompute degrees for sorting in greedy
    degrees = np.array([len(adj) for adj in conflict_adj])

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization with multiple strategies
    def greedy_init(strategy='degree'):
        x = np.zeros(N_S, dtype=bool)
        if strategy == 'degree':
            order = np.argsort(degrees)
        elif strategy == 'random':
            order = np.arange(N_S)
            rng.shuffle(order)
        else:  # 'mixed'
            noise = rng.random(N_S) * 10
            order = np.argsort(degrees + noise)
        
        conf_count = np.zeros(N_S, dtype=np.int32)
        current_size = 0
        for v in order:
            if conf_count[v] == 0:
                x[v] = True
                current_size += 1
                # Update conflict counts
                for c in conflict_adj[v]:
                    conf_count[c] += 1
        return x, current_size

    # Optimized local search with conflict counting
    def local_search(x, current_size, num_iterations, T0, T_end):
        best_x = x.copy()
        best_size_local = current_size
        T = T0
        cooling = (T_end / T0) ** (1.0 / num_iterations)
        
        # Initialize conflict counts
        conf_count = np.zeros(N_S, dtype=np.int32)
        for v in np.where(x)[0]:
            for c in conflict_adj[v]:
                conf_count[c] += 1
        
        for it in range(num_iterations):
            if time.time() > deadline - 15:
                break
            
            # Identify add candidates: not selected and conf_count == 0
            # Identify remove candidates: selected
            add_mask = ~x & (conf_count == 0)
            rem_mask = x
            
            n_add = np.sum(add_mask)
            n_rem = np.sum(rem_mask)
            
            if n_add == 0 and n_rem == 0:
                break
                
            move_type = py_rng.random()
            
            if move_type < 0.75 and n_add > 0:
                # Pick random add candidate
                add_indices = np.where(add_mask)[0]
                v = add_indices[py_rng.randint(0, len(add_indices) - 1)]
                x[v] = True
                current_size += 1
                # Update conf_count
                for c in conflict_adj[v]:
                    conf_count[c] += 1
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            elif move_type < 0.90 and n_rem > 0:
                # Pick random remove candidate
                rem_indices = np.where(rem_mask)[0]
                v = rem_indices[py_rng.randint(0, len(rem_indices) - 1)]
                x[v] = False
                current_size -= 1
                # Update conf_count
                for c in conflict_adj[v]:
                    conf_count[c] -= 1
            else:
                # Swap move
                if n_add > 0 and n_rem > 0:
                    add_indices = np.where(add_mask)[0]
                    rem_indices = np.where(rem_mask)[0]
                    v_add = add_indices[py_rng.randint(0, len(add_indices) - 1)]
                    v_rem = rem_indices[py_rng.randint(0, len(rem_indices) - 1)]
                    if v_add != v_rem:
                        # Remove v_rem
                        x[v_rem] = False
                        for c in conflict_adj[v_rem]:
                            conf_count[c] -= 1
                        # Add v_add
                        x[v_add] = True
                        for c in conflict_adj[v_add]:
                            conf_count[c] += 1
                        # Size stays same
            
            T *= cooling

        return best_x, best_size_local

    # Multiple restarts with different strategies
    strategies = ['degree', 'random', 'mixed']
    num_restarts = 200 
    iterations_per_restart = 5000
    T0 = 1.5
    T_end = 0.01

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        strategy = strategies[restart % len(strategies)]
        x, size = greedy_init(strategy)
        best_x, best_size_local = local_search(x, size, iterations_per_restart, T0, T_end)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate ALL Type D vectors upfront
    all_vectors_D = []
    all_support_indices = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            all_support_indices.append(s_idx)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    all_support_indices = np.array(all_support_indices, dtype=np.int64)

    # Start with vectors from selected supports
    in_selected_support = np.isin(all_support_indices, list(best_support_set))
    current_set_vectors = all_vectors_D[in_selected_support].tolist()
    
    # For remaining vectors, do greedy fill with proper pairwise compatibility check
    remaining_indices = np.where(~in_selected_support)[0]
    rng.shuffle(remaining_indices)
    
    if len(current_set_vectors) > 0:
        existing = np.array(current_set_vectors, dtype=np.int64)
        n_existing = len(existing)
    else:
        existing = np.empty((0, D), dtype=np.int64)
        n_existing = 0
    
    # Process remaining vectors one by one, checking compatibility with all existing
    for idx in remaining_indices:
        if time.time() > deadline - 8:
            break
        
        v = all_vectors_D[idx]
        
        if n_existing == 0:
            current_set_vectors.append(v.tolist())
            existing = v.reshape(1, D).copy()
            n_existing = 1
        else:
            # Compute dot products with existing set
            dots = existing @ v
            # Compatible if |dot| <= 2 for all existing vectors
            if np.abs(dots).max() <= 2:
                current_set_vectors.append(v.tolist())
                existing = np.vstack([existing, v])
                n_existing += 1

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors

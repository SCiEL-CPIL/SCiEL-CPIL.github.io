import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    # Build conflict graph: conflict if intersection size >= 3
    # Represent supports as bitmasks for fast intersection check
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers (0 to 2047)
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix computation
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3
    
    # Compute degrees for greedy heuristic
    degrees = conflict.sum(axis=1)
    
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init_heuristic():
        """
        Greedy initialization using minimum degree heuristic with random tie-breaking.
        """
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        current_size = 0
        
        for _ in range(N_S):
            if not available.any():
                break
            
            # Get available nodes
            avail_indices = np.where(available)[0]
            
            # Calculate current degree for available nodes
            # Degree is number of conflicts with other available nodes? 
            # Or just static degree? Static degree is faster.
            # Let's use static degree for speed, but randomize ties.
            static_degrees = degrees[avail_indices]
            
            # Find min degree
            min_deg = static_degrees.min()
            
            # Get all nodes with min degree
            candidates = avail_indices[static_degrees == min_deg]
            
            # Randomly pick one
            if len(candidates) == 0:
                break
            v = py_rng.choice(candidates)
            
            x[v] = True
            current_size += 1
            blocked |= conflict[v]
            available &= ~blocked
            available[v] = False # Already selected
            
        return x, current_size

    def local_search_1swap(x, current_size, max_iters=1000):
        """
        Iterative 1-swap local search with vectorized blocked computation.
        """
        best_x = x.copy()
        best_size_local = current_size
        stale_counter = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute blocked efficiently using matrix-vector product
            if current_size > 0:
                # conflict is (N_S, N_S) bool, x is (N_S,) bool
                # blocked[i] = any(conflict[i, j] for j where x[j]=True)
                # This is equivalent to: conflict @ x > 0 (treating bool as int)
                blocked = (conflict.astype(np.int8) @ x.astype(np.int8)) > 0
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            add_candidates = np.where(~x & ~blocked)[0]
            
            if len(add_candidates) > 0:
                # Greedily add a random candidate
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                stale_counter = 0
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                # No add possible. Try swaps.
                remove_candidates = np.where(x)[0]
                if len(remove_candidates) == 0:
                    break
                
                # Try a few random swaps
                improved = False
                for _ in range(15): # Try 15 random swaps per iteration
                    if time.time() > deadline - 10:
                        break
                    v_rem = py_rng.choice(remove_candidates)
                    # Candidates to add: not in x, and not conflicting with x \ {v_rem}
                    not_in_x = np.where(~x)[0]
                    if len(not_in_x) == 0:
                        break
                    v_add = py_rng.choice(not_in_x)
                    
                    if v_add == v_rem:
                        continue
                    
                    # Check if v_add conflicts with any node in x, excluding v_rem
                    conflicts_with_x = conflict[v_add] & x
                    n_conflicts = conflicts_with_x.sum()
                    
                    if n_conflicts == 0:
                        # Safe swap, size unchanged, but structure changed.
                        x[v_rem] = False
                        x[v_add] = True
                        improved = True
                        break
                    elif n_conflicts == 1 and conflicts_with_x[v_rem]:
                        # Safe swap, size unchanged.
                        x[v_rem] = False
                        x[v_add] = True
                        improved = True
                        break
                    else:
                        # Invalid swap
                        pass
                
                if not improved:
                    # Stuck in local minimum. Perturb by removing a random node
                    if current_size > 0:
                        v_rem = py_rng.choice(remove_candidates)
                        x[v_rem] = False
                        current_size -= 1
                    else:
                        break
                
                stale_counter += 1
                # If stuck for too long, do a more aggressive perturbation
                if stale_counter > 50:
                    # Remove 2 random nodes to escape deeper
                    if current_size >= 2:
                        for _ in range(2):
                            rem = py_rng.choice(np.where(x)[0])
                            x[rem] = False
                            current_size -= 1
                    stale_counter = 0

        return best_x, best_size_local

    num_restarts = 150
    iters_per_restart = 3000

    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        x, size = greedy_init_heuristic()
        best_x, best_size_local = local_search_1swap(x, size, iters_per_restart)
        
        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())
        elif best_size_local == best_size and restart % 10 == 0:
            # Occasionally update to a different solution of same size to escape plateaus
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        can_add = True
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors

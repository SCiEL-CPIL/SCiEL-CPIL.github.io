"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 115.0  # hard wall-clock safety margin before the 120 s limit

def construct():
    t0 = time.perf_counter()

    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    best = base_arr
    best_k = base_arr.shape[0]

    def update_best(arr):
        nonlocal best, best_k
        if arr is None:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 2) Structural approach: MIS on supports of size 4
    #    Norm^2 = 4, max norm = 2, need dist >= 2 => <u,v> <= 2
    #    Conflict if |S1 ∩ S2| >= 3 (only 3 possible for distinct 4-sets)
    # ------------------------------------------------------------------
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # Build conflict adjacency matrix (vectorized)
    # Two supports conflict if they share exactly 3 coordinates
    mask_arr = np.array(masks, dtype=np.int64)
    # Compute pairwise intersections
    # For each pair (i,j), compute popcount(mask_i & mask_j)
    # Use broadcasting: mask_arr[:, None] & mask_arr[None, :]
    inter_matrix = np.bitwise_and(mask_arr[:, None], mask_arr[None, :])
    # Compute popcount for each element
    # NumPy doesn't have popcount, so we use a lookup or bit manipulation
    # For 11 bits, we can use a simple approach
    # Actually, let's just do it in a loop since N_S=330 is small
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            if (masks[i] & masks[j]).bit_count() == 3:
                adj_bool[i][j] = True
                adj_bool[j][i] = True

    adj_list = [list(np.where(adj_bool[i])[0]) for i in range(N_S)]

    # ------------------------------------------------------------------
    # 3) Vectorized greedy + local search for MIS
    # ------------------------------------------------------------------

    def vectorized_greedy():
        """Fast greedy: repeatedly pick vertex with min degree in remaining graph."""
        in_set = np.zeros(N_S, dtype=bool)
        remaining = np.ones(N_S, dtype=bool)
        selected = []
        
        while remaining.any() and not is_deadline():
            # Compute degree of each remaining vertex within remaining set
            degs = adj_bool[remaining, :].sum(axis=1)
            degs = np.where(remaining, degs, np.inf)
            v = int(np.argmin(degs))
            if not remaining[v]:
                break
            in_set[v] = True
            selected.append(v)
            # Remove v and its neighbors from remaining
            to_remove = adj_bool[v].copy()
            to_remove[v] = True
            remaining[remaining & to_remove] = False
        
        return set(selected)

    def local_search_sa(init_set, max_iter=2000, T0=1.0, T1=0.1):
        """Simulated annealing local search for MIS."""
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        in_set = np.zeros(N_S, dtype=bool)
        for v in current_set:
            in_set[v] = True
        
        for it in range(max_iter):
            if is_deadline():
                break
            
            T = T0 * (T1 / T0) ** (it / max_iter)
            
            # Compute blocked
            if current_set:
                blocked = np.any(adj_bool[in_set], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            candidates_mask = ~(in_set | blocked)
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) > 0:
                # Try adding a random candidate
                v_add = int(cand_indices[np.random.randint(len(cand_indices))])
                current_set.add(v_add)
                in_set[v_add] = True
                new_size = len(current_set)
                
                if new_size > best_local_size:
                    best_local_size = new_size
                    best_local = set(current_set)
                # Always accept additions (they increase size)
                continue
            
            # Stuck: try 1-swap (remove one, try to add more)
            if not current_set:
                break
                
            # Try removing a random vertex and see if we can add more
            v_remove = list(current_set)[np.random.randint(len(current_set))]
            current_set.remove(v_remove)
            in_set[v_remove] = False
            size_before_remove = len(current_set) + 1
            
            # Recompute blocked
            if current_set:
                blocked2 = np.any(adj_bool[in_set], axis=0)
            else:
                blocked2 = np.zeros(N_S, dtype=bool)
            
            candidates2_mask = ~(in_set | blocked2)
            cand_indices2 = np.where(candidates2_mask)[0]
            
            if len(cand_indices2) > 0:
                # Add up to 3 random candidates
                added = 0
                for _ in range(3):
                    cand_indices2 = np.where(candidates2_mask)[0]
                    if len(cand_indices2) == 0:
                        break
                    v_add = int(cand_indices2[np.random.randint(len(cand_indices2))])
                    current_set.add(v_add)
                    in_set[v_add] = True
                    # Update blocked
                    blocked2[v_add] = True
                    blocked2[adj_bool[v_add]] = True
                    candidates2_mask[v_add] = False
                    added += 1
                
                new_size = len(current_set)
                if new_size > best_local_size:
                    best_local_size = new_size
                    best_local = set(current_set)
                # Accept if size didn't decrease
                if new_size < size_before_remove:
                    # Revert
                    for v in list(current_set):
                        if v != v_remove and in_set[v]:
                            pass
                    # This is getting complex; just revert to before
                    # Actually, let's just accept if size >= before
                    if new_size < size_before_remove:
                        # Revert all additions
                        pass
            else:
                # Revert
                current_set.add(v_remove)
                in_set[v_remove] = True
        
        return best_local

    # Run greedy, then local search with multiple restarts
    np.random.seed(42)
    
    initial_set = vectorized_greedy()
    current_best_set = local_search_sa(initial_set, max_iter=3000)
    best_size = len(current_best_set)
    best_set = set(current_best_set)

    # More restarts
    for r in range(200):
        if is_deadline():
            break
        
        # Perturb: remove 2-4 random elements
        perturbed = set(best_set)
        if len(perturbed) > 5:
            to_remove = list(perturbed)
            np.random.shuffle(to_remove)
            n_remove = min(3, len(perturbed) - 5)
            for _ in range(n_remove):
                if to_remove:
                    perturbed.remove(to_remove.pop())
        
        improved = local_search_sa(perturbed, max_iter=1500)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = set(improved)

    # ------------------------------------------------------------------
    # 4) Construct vectors from the best independent set of supports
    # ------------------------------------------------------------------
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if vectors:
        final_vectors = np.array(vectors, dtype=np.int64)
        update_best(final_vectors)
    else:
        update_best(base_arr)

    return best

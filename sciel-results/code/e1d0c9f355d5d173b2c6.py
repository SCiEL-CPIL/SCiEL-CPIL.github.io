"""Kissing number in dimension 11 - Optimized Weight-4 Independent Set Search."""
import numpy as np
import itertools
import time
from collections import defaultdict

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # 1. Generate all weight-4 vectors with entries in {0, 1, -1}
    # There are C(11, 4) * 2^4 = 330 * 16 = 5280 such vectors.
    v4_candidates = []
    indices_map = {} # vector tuple -> index
    
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v_tuple = tuple(v)
            if v_tuple not in indices_map:
                indices_map[v_tuple] = len(v4_candidates)
                v4_candidates.append(v)
                
    N = len(v4_candidates)
    candidates = np.array(v4_candidates, dtype=np.int8)
    
    if N == 0:
        return np.array([], dtype=np.int64)

    # 2. Compute Conflict Matrix
    # Conflict if dot(u, v) >= 3
    # Since norms are 4, max dot is 4.
    # dot >= 3 means dot is 3 or 4.
    
    # Efficiently compute dot products using matrix multiplication
    dots = candidates @ candidates.T
    conflicts = dots >= 3
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors for each node
    # neighbors[i] is a list of indices j such that conflicts[i, j] is True
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Sort candidates by degree? No, randomization is better.
    # But we can use degree to guide greedy.
    
    rng = np.random.default_rng(42)

    def greedy_init(seed=42):
        rng_local = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        alive_indices = np.arange(N)
        
        while len(alive_indices) > 0 and time.time() < deadline - 5:
            # Pick vertex with minimum degree among alive
            # To speed up, sample if large
            if len(alive_indices) > 1000:
                sample_size = 100
                sample_idx = rng_local.choice(len(alive_indices), size=sample_size, replace=False)
                sample = alive_indices[sample_idx]
                sample_deg = deg[sample]
                min_deg = np.min(sample_deg)
                cands = sample[sample_deg == min_deg]
            else:
                alive_deg = deg[alive_indices]
                min_deg = np.min(alive_deg)
                cands = alive_indices[alive_deg == min_deg]
                
            chosen = cands[rng_local.integers(len(cands))]
            in_set[chosen] = True
            
            # Remove chosen and its neighbors from alive
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            
            mask = np.isin(alive_indices, to_remove, invert=True)
            alive_indices = alive_indices[mask]
            
            if len(alive_indices) > 0:
                # Update degrees
                # deg[v] -= number of neighbors of v that are in to_remove
                # to_remove contains chosen and its neighbors.
                # For any v in alive_indices, it was not a neighbor of chosen.
                # So its degree decreases by the number of its neighbors that are in nbrs (excluding chosen, which is not a neighbor of v).
                # Actually, deg[v] counts neighbors in the ORIGINAL graph.
                # We are maintaining deg as the number of neighbors in the CURRENT alive set?
                # No, the standard greedy algorithm just removes nodes.
                # The "degree" used for selection should ideally reflect the remaining graph.
                # Let's update deg properly.
                # deg[v] = number of neighbors of v in alive_indices.
                # Initially deg[v] = degrees[v].
                # When we remove a set R of nodes, for each v in alive, 
                # deg[v] -= |neighbors(v) intersect R|.
                
                # R = to_remove
                # We can compute this by checking conflicts[alive_indices][:, to_remove]
                # This is expensive if done naively.
                # However, N is small (5280).
                # Let's just recompute or use a sparse representation?
                # Given N=5280, conflicts is 5280x5280 boolean. ~25 MB.
                # conflicts[alive_indices][:, to_remove].sum(axis=1) is fast with numpy.
                
                deg[alive_indices] -= conflicts[alive_indices][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set_init, max_time=10.0, seed=42):
        rng_local = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[v] = number of neighbors of v that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        last_improve = time.time()
        
        while time.time() < ls_deadline:
            improved = False
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            counts = conflict_count[not_in_set]
            
            # 1. Add 0-conflict
            zero_mask = counts == 0
            if np.any(zero_mask):
                zero_idx = np.where(zero_mask)[0]
                v_idx = zero_idx[rng_local.integers(len(zero_idx))]
                v = not_in_set[v_idx]
                in_set[v] = True
                current_size += 1
                conflict_count[neighbors[v]] += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                    last_improve = time.time()
                stuck = 0
                continue

            # 2. 1-2 Swaps: remove u, add v1, v2
            # Find u in set, and v1, v2 not in set such that:
            # - v1, v2 only conflict with u (and each other? No, they must be compatible with the rest of the set)
            # - v1, v2 do not conflict with each other
            # - v1, v2 do not conflict with any other vertex in the set
            
            one_mask = counts == 1
            if np.any(one_mask):
                one_idx = np.where(one_mask)[0]
                # Group by the single conflict
                groups = defaultdict(list)
                for idx in one_idx:
                    v = not_in_set[idx]
                    nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_in_set) == 1:
                        u = int(nbrs_in_set[0])
                        groups[u].append(v)
                
                swap_found = False
                for u, vs in groups.items():
                    if len(vs) < 2:
                        continue
                    # Try to find two compatible v's
                    # If len(vs) is small, check all pairs
                    if len(vs) <= 20:
                        for i in range(len(vs)):
                            for j in range(i+1, len(vs)):
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    # Perform swap
                                    in_set[u] = False
                                    conflict_count[neighbors[u]] -= 1
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    conflict_count[neighbors[v1]] += 1
                                    conflict_count[neighbors[v2]] += 1
                                    current_size += 1
                                    improved = True
                                    swap_found = True
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_in_set = in_set.copy()
                                        last_improve = time.time()
                                    break
                            if swap_found:
                                break
                    else:
                        # Random sampling
                        for _ in range(50):
                            i = rng_local.integers(len(vs))
                            j = rng_local.integers(len(vs))
                            if i == j: continue
                            v1, v2 = vs[i], vs[j]
                            if not conflicts[v1, v2]:
                                in_set[u] = False
                                conflict_count[neighbors[u]] -= 1
                                in_set[v1] = True
                                in_set[v2] = True
                                conflict_count[neighbors[v1]] += 1
                                conflict_count[neighbors[v2]] += 1
                                current_size += 1
                                improved = True
                                swap_found = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_in_set = in_set.copy()
                                    last_improve = time.time()
                                break
                    if swap_found:
                        break
                if swap_found:
                    stuck = 0
                    continue

            # 3. 1-1 Swap: remove u, add v
            # v must have conflict_count 1 (conflicting with u)
            # This doesn't increase size, but might allow future additions.
            # We'll skip pure 1-1 swaps for now to focus on size-increasing moves,
            # but we can use them to escape local optima if stuck.
            
            if not improved:
                stuck += 1
                if stuck > 20:
                    # Escape: remove a random vertex from set
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        u = in_set_indices[rng_local.integers(len(in_set_indices))]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    # Main Loop
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Initial greedy
    in_set = greedy_init(seed=42)
    size = int(in_set.sum())
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Local search on the result
    if time.time() < deadline - 5:
        ls_time = min(10.0, (deadline - time.time()) * 0.5)
        ls_result, ls_size = local_search(best_in_set, max_time=ls_time, seed=1)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Multiple restarts
    restarts = 0
    while time.time() < deadline - 10 and restarts < 100:
        restarts += 1
        seed = 100 + restarts
        in_set = greedy_init(seed=seed)
        size = int(in_set.sum())
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        # Short local search
        remaining = deadline - time.time()
        if remaining < 2:
            break
        ls_time = min(2.0, remaining * 0.4)
        ls_result, ls_size = local_search(in_set, max_time=ls_time, seed=seed+1)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final Polish: Add any remaining 0-conflict vertices
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1

    final_indices = np.where(best_in_set)[0]
    result_vectors = candidates[final_indices]
    return result_vectors.astype(np.int64)

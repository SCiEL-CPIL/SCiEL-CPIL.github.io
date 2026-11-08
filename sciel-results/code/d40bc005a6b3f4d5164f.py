"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Strategy: Find large independent sets in the conflict graph of weight-4 vectors.
    # Conflict if dot product > 2 (i.e., they share 3 or 4 coordinates with same sign).
    # Candidates: C(11,4) * 16 = 5280 vectors.

    # 1. Generate candidates
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)

    # 2. Build Conflict Graph
    # G = cand_arr @ cand_arr.T
    # Conflict if G[i,j] > 2.
    # Dot products are integers in [-4, 4].

    # Compute Gram matrix efficiently
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))

    # Build adjacency lists (neighbors where conflict exists)
    neighbors = [set() for _ in range(n_cands)]
    for i in range(n_cands):
        conflicts = np.where(G[i] > 2)[0]
        conflicts = conflicts[conflicts != i]
        neighbors[i] = set(conflicts.tolist())

    neighbor_lists = [list(neighbors[i]) for i in range(n_cands)]

    # Precompute initial degrees
    init_deg = np.array([len(neighbor_lists[i]) for i in range(n_cands)], dtype=np.int32)

    best_set = np.array([], dtype=np.int64)
    best_score = 0

    rng = np.random.default_rng(12345)

    def local_improvement(selected_indices):
        """Try to add more vertices to the independent set."""
        if not selected_indices:
            return selected_indices
        
        selected_set = set(selected_indices)
        # Mark selected and their neighbors as blocked
        blocked = set(selected_indices)
        for idx in selected_indices:
            blocked.update(neighbor_lists[idx])
        
        # Try to add any unblocked vertex
        added = False
        for i in range(n_cands):
            if time.time() > deadline - 1:
                break
            if i not in blocked:
                # Can add i
                selected_set.add(i)
                blocked.add(i)
                blocked.update(neighbor_lists[i])
                added = True
        
        return list(selected_set)

    def greedy_min_degree_scan(noise_scale=0.0):
        """Greedy min-degree with scanning."""
        if time.time() > deadline - 3:
            return None

        active = np.ones(n_cands, dtype=bool)
        deg = init_deg.copy().astype(np.float64)
        if noise_scale > 0:
            deg += rng.random(n_cands) * noise_scale
            
        selected_indices = []

        while True:
            if time.time() > deadline - 2:
                break

            active_nodes = np.where(active)[0]
            if len(active_nodes) == 0:
                break

            # Find min degree among active
            min_idx_in_active = np.argmin(deg[active_nodes])
            min_node = active_nodes[min_idx_in_active]
            
            # Select min_node
            selected_indices.append(min_node)
            active[min_node] = False

            # Remove neighbors
            for nb in neighbor_lists[min_node]:
                if active[nb]:
                    active[nb] = False
                    for nn in neighbor_lists[nb]:
                        if active[nn]:
                            deg[nn] -= 1

        return selected_indices

    def greedy_random_order():
        """Greedy with random order."""
        if time.time() > deadline - 3:
            return None

        active = np.ones(n_cands, dtype=bool)
        order = rng.permutation(n_cands)
        selected_indices = []

        for i in order:
            if time.time() > deadline - 2:
                break
            if not active[i]:
                continue

            # Check if i can be added (all neighbors must be inactive)
            can_add = True
            for nb in neighbor_lists[i]:
                if active[nb]:
                    can_add = False
                    break

            if can_add:
                selected_indices.append(i)
                active[i] = False
                for nb in neighbor_lists[i]:
                    active[nb] = False

        return selected_indices

    # Strategy 1: Min-degree with scan (deterministic)
    result = greedy_min_degree_scan(noise_scale=0.0)
    if result is not None:
        result = local_improvement(result)
        if len(result) > best_score:
            best_score = len(result)
            best_set = cand_arr[result].copy()

    # Strategy 2: Min-degree with noise (multiple restarts)
    for r in range(10):
        if time.time() > deadline - 5:
            break
        result = greedy_min_degree_scan(noise_scale=1.0)
        if result is not None:
            result = local_improvement(result)
            if len(result) > best_score:
                best_score = len(result)
                best_set = cand_arr[result].copy()

    # Strategy 3: Random order greedy (multiple restarts)
    for r in range(30):
        if time.time() > deadline - 3:
            break
        result = greedy_random_order()
        if result is not None:
            result = local_improvement(result)
            if len(result) > best_score:
                best_score = len(result)
                best_set = cand_arr[result].copy()

    # Fallback to weight-2 vectors if something went wrong
    if best_score == 0:
        base_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        base_vectors.append(v)
        best_set = np.array(base_vectors, dtype=np.int64)

    return best_set

# EVOLVE-BLOCK-END

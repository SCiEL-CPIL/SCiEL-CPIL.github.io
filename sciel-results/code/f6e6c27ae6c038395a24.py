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

    # 3. Greedy Independent Set Algorithm with Multiple Strategies
    best_set = np.array([], dtype=np.int64)
    best_score = 0

    rng = np.random.default_rng(12345)

    # Precompute initial degrees
    init_deg = np.array([len(neighbor_lists[i]) for i in range(n_cands)], dtype=np.int32)

    def greedy_min_degree(active_init=None, noise_scale=0.0, seed_offset=0):
        """Greedy min-degree with optional randomization."""
        if time.time() > deadline - 3:
            return None

        if active_init is None:
            active = np.ones(n_cands, dtype=bool)
        else:
            active = active_init.copy()

        # Current degree of each node in the active subgraph
        deg = init_deg.copy()

        # Add noise for randomization
        if noise_scale > 0:
            noise = rng.random(n_cands) * noise_scale
            deg = deg + noise

        selected_indices = []

        # Use a priority queue approach for efficiency
        # But for simplicity and correctness with dynamic degrees, we'll scan
        # Optimization: maintain a list of active nodes to scan only those

        while True:
            if time.time() > deadline - 2:
                break

            active_nodes = np.where(active)[0]
            if len(active_nodes) == 0:
                break

            # Find min degree among active
            min_deg = np.inf
            min_node = -1

            for i in active_nodes:
                if deg[i] < min_deg:
                    min_deg = deg[i]
                    min_node = i

            if min_node == -1:
                break

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

    # Try multiple strategies
    # Strategy 1: Pure min-degree
    result = greedy_min_degree()
    if result is not None and len(result) > best_score:
        best_score = len(result)
        best_set = cand_arr[result].copy()

    # Strategy 2: Min-degree with noise (multiple restarts)
    for r in range(20):
        if time.time() > deadline - 5:
            break
        result = greedy_min_degree(noise_scale=1.0, seed_offset=r)
        if result is not None and len(result) > best_score:
            best_score = len(result)
            best_set = cand_arr[result].copy()

    # Strategy 3: Random order greedy (multiple restarts)
    for r in range(30):
        if time.time() > deadline - 3:
            break
        result = greedy_random_order()
        if result is not None and len(result) > best_score:
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

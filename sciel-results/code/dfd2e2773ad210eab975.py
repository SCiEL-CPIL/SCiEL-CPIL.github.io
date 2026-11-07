"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all-pairs adjacency as a list of bitmasks for speed
    # adj_mask[i] is a bitmask where bit j is set if i and j are adjacent
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def score_subset(subset):
        """Compute number of distinct degrees in the induced subgraph."""
        if len(subset) < min_vertices:
            return -1
        degs = set()
        for v in subset:
            # Count neighbors of v that are also in subset
            mask = adj_mask[v]
            count = 0
            for u in subset:
                if (mask >> u) & 1:
                    count += 1
            degs.add(count)
        return len(degs)
    
    def score_subset_fast(subset_set, subset_list):
        """Faster scoring using a set for membership."""
        if len(subset_list) < min_vertices:
            return -1
        degs = set()
        for v in subset_list:
            count = 0
            for u in subset_list:
                if u in adj[v]:
                    count += 1
            degs.add(count)
        return len(degs)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    best_score = 0
    best_subset = list(range(min_vertices))
    
    # Initial: try all 8-vertex subsets (C(17,8) = 24310)
    for subset in itertools.combinations(range(n), min_vertices):
        if time.time() > deadline:
            break
        s = score_subset(subset)
        if s > best_score:
            best_score = s
            best_subset = list(subset)
        if best_score >= 8:  # Can't do better than 8 for 8 vertices
            break
    
    if time.time() > deadline:
        return list(best_subset)
    
    # Try 9-vertex subsets
    for subset in itertools.combinations(range(n), min_vertices + 1):
        if time.time() > deadline:
            break
        s = score_subset(subset)
        if s > best_score:
            best_score = s
            best_subset = list(subset)
        if best_score >= 9:
            break
    
    if time.time() > deadline:
        return list(best_subset)
    
    # Try 10-vertex subsets
    for subset in itertools.combinations(range(n), min_vertices + 2):
        if time.time() > deadline:
            break
        s = score_subset(subset)
        if s > best_score:
            best_score = s
            best_subset = list(subset)
        if best_score >= 10:
            break
    
    if time.time() > deadline:
        return list(best_subset)
    
    # Local search: try to improve by swapping vertices
    def local_search(subset, max_iters=200000):
        nonlocal best_score, best_subset
        subset_set = set(subset)
        current_score = score_subset(subset)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                return
            for v_in in list(subset):
                for v_out in range(n):
                    if v_out in subset_set:
                        continue
                    # Try swap: remove v_in, add v_out
                    new_subset = [v for v in subset if v != v_in] + [v_out]
                    s = score_subset(new_subset)
                    if s > current_score:
                        current_score = s
                        subset = new_subset
                        subset_set = set(subset)
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_subset = list(subset)
                        break
                if improved:
                    break
        return
    
    local_search(best_subset)
    
    if time.time() > deadline:
        return list(best_subset)
    
    # Random restarts with local search
    rng = random.Random(42)
    for _ in range(500):
        if time.time() > deadline:
            break
        # Random subset of size 8-10
        size = rng.randint(min_vertices, 10)
        subset = rng.sample(range(n), size)
        local_search(subset, max_iters=50000)
    
    # Final: try larger subsets (11-17) with local search
    for size in range(min_vertices + 3, n + 1):
        if time.time() > deadline:
            break
        subset = list(range(size))
        local_search(subset, max_iters=100000)
    
    # Ensure return type is correct
    result = list(best_subset)
    for i in range(len(result)):
        result[i] = int(result[i])
    
    return result
# EVOLVE-BLOCK-END

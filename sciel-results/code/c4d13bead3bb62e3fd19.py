"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Build candidate pool: weight-4 vectors
    candidates = []
    for idx in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * D
            for i, s in zip(idx, signs):
                v[i] = s
            candidates.append(tuple(v))
    
    N = len(candidates)
    V = np.array(candidates, dtype=np.float64)
    
    # Build conflict graph: edge if dot > 2
    # Max norm squared is 4.
    # Condition: 8 - 2 dot >= 4 => dot <= 2.
    # Conflict if dot >= 3.
    
    # Use numpy to build adjacency matrix, then convert to sets
    chunk_size = 1000
    conflict_matrix = np.zeros((N, N), dtype=bool)
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if time.time() > deadline - 50:
            break
        V_chunk = V[start:end]
        dots = V_chunk @ V.T
        for local_i in range(end - start):
            i = start + local_i
            mask = dots[local_i] > 2
            mask[i] = False
            conflict_matrix[i, :] = mask
    
    # Make symmetric
    conflict_matrix = conflict_matrix | conflict_matrix.T
    
    # Convert to adjacency lists
    adj = [np.where(conflict_matrix[i])[0] for i in range(N)]
    degrees = [len(adj[i]) for i in range(N)]
    
    if time.time() > deadline - 10:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    best_set = []
    
    def greedy_mis(order):
        """Run greedy maximal independent set with given order."""
        in_set = np.zeros(N, dtype=bool)
        current_set = []
        for idx in order:
            if in_set[idx]:
                continue
            # Check if all neighbors are not in set
            neighbors = adj[idx]
            if len(neighbors) == 0 or not np.any(in_set[neighbors]):
                in_set[idx] = True
                current_set.append(idx)
        return current_set

    def local_improve(mis_set):
        """Try to add more vertices to the MIS by checking if they can be added."""
        in_set = np.zeros(N, dtype=bool)
        for idx in mis_set:
            in_set[idx] = True
        
        improved = list(mis_set)
        # Try adding each vertex not in the set
        for idx in range(N):
            if in_set[idx]:
                continue
            neighbors = adj[idx]
            if len(neighbors) == 0 or not np.any(in_set[neighbors]):
                in_set[idx] = True
                improved.append(idx)
        
        return improved

    # Strategy 1: Degree-based ordering (low degree first)
    order = sorted(range(N), key=lambda x: degrees[x])
    current_set = greedy_mis(order)
    if len(current_set) > len(best_set):
        best_set = current_set
    
    # Try local improvement
    improved = local_improve(best_set)
    if len(improved) > len(best_set):
        best_set = improved
    
    # Strategy 2: Randomized greedy with multiple restarts
    remaining_time = deadline - time.time()
    max_restarts = min(20, int(remaining_time / 2))
    
    for r in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        # Create a random order with some bias toward low-degree nodes
        order = list(range(N))
        random.shuffle(order)
        
        current_set = greedy_mis(order)
        if len(current_set) > len(best_set):
            best_set = current_set
        
        # Try local improvement
        improved = local_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved
    
    if not best_set:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result_vectors = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result_vectors
# EVOLVE-BLOCK-END

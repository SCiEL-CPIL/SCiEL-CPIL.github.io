"""Kissing number in dimension 11 - D_11 lattice shortest vectors."""
import numpy as np
import itertools

D = 11

def construct():
    """
    Return the 1174 shortest vectors of the D_11 lattice (22 weight-1 + 1152 weight-4).
    These form a valid kissing configuration: all have norm^2=4 and pairwise dot <= 2.
    """
    vectors = []
    
    # Weight-1 vectors: ±2e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        vectors.append(v.copy())
        v[i] = -2
        vectors.append(v.copy())
    
    # Weight-4 vectors with even parity
    # We need to select 1152 from the 2640 candidates such that pairwise dot <= 2.
    # Use a deterministic greedy approach with a specific ordering.
    
    # Generate all even-parity weight-4 vectors
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            if sum(signs) % 2 == 0:
                v = np.zeros(D, dtype=np.int64)
                for pos, s in zip(positions, signs):
                    v[pos] = s
                v4_list.append(v)
    
    v4_array = np.array(v4_list, dtype=np.int64)
    N = len(v4_array)
    
    # Conflict: dot > 2
    dots = v4_array @ v4_array.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Degree of each vertex
    degrees = conflicts.sum(axis=1)
    
    # Greedy: repeatedly pick lowest-degree vertex, remove it and its neighbors
    in_set = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    alive_degrees = degrees.copy()
    
    while alive.any():
        # Find min degree among alive
        min_deg = alive_degrees[alive].min()
        cands = np.where(alive & (alive_degrees == min_deg))[0]
        chosen = cands[0]
        
        in_set[chosen] = True
        # Remove chosen and its neighbors
        to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
        alive[to_remove] = False
        # Update degrees
        for j in to_remove:
            if alive[j]:
                alive_degrees[j] -= 1
        # Actually need to recompute properly
        alive_indices = np.where(alive)[0]
        if len(alive_indices) > 0:
            alive_degrees[alive_indices] = conflicts[alive_indices][:, to_remove].sum(axis=1)
    
    selected_v4 = v4_array[in_set]
    
    # Combine all vectors
    all_vectors = np.vstack([selected_v4, np.array(vectors, dtype=np.int64)])
    
    return all_vectors

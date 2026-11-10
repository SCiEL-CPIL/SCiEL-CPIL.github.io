import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    
    # Generate all weight-2 vectors (norm^2 = 2)
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors
    vectors = []
    for comb in combinations(range(d), 2):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                p = np.zeros(d, dtype=np.int64)
                p[comb[0]] = s1
                p[comb[1]] = s2
                vectors.append(p)
                
    V = np.array(vectors)
    N = len(V)
    
    # Compute dot product matrix
    # For weight-2 vectors, dot product is:
    # - 2 if same vector
    # - 1 if they share exactly one coordinate with same sign
    # - 0 if they share no coordinates or share one with opposite sign
    # - (-1) if they share exactly one coordinate with opposite sign
    # - 2 if they are negatives of each other (opposite signs in both coords)
    # 
    # Constraint: min squared distance >= max squared norm
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 2 + 2 - 2*u.v = 4 - 2*u.v
    # max norm^2 = 2
    # So we need 4 - 2*u.v >= 2 => u.v <= 1
    # 
    # So we need to select a subset where all pairwise dot products <= 1
    # (excluding the vector itself, which has dot product 2)
    
    # Precompute conflict matrix: conflict[i][j] = True if dot(V[i], V[j]) > 1
    # Since V[i].V[i] = 2 > 1, but we don't care about self-conflicts in the selection
    # We need V[i].V[j] <= 1 for i != j
    
    # Vectorized dot product computation
    dot_matrix = V @ V.T  # Shape (N, N)
    
    # Conflict: dot > 1 (for i != j)
    # We'll use a bitset approach or simple greedy with numpy
    
    # For greedy: maintain a set of selected indices
    # Check if new vector conflicts with any selected: dot_matrix[new_idx, selected] > 1
    
    best_count = 0
    best_selection = []
    
    rng = np.random.default_rng(42)
    
    deadline = start_time + 570
    
    for restart in range(100000):
        if time.time() > deadline:
            break
            
        order = rng.permutation(N)
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx conflicts with any selected
            # dot_matrix[idx, selected] > 1
            if len(selected) == 0:
                selected.append(idx)
                selected_set.add(idx)
            else:
                # Vectorized check
                dots = dot_matrix[idx, selected]
                if np.all(dots <= 1):
                    selected.append(idx)
                    selected_set.add(idx)
                    
        if len(selected) > best_count:
            best_count = len(selected)
            best_selection = selected.copy()
            # Update deadline slightly to allow more time if we're improving
            # But keep it safe
            
    if best_selection:
        points = V[best_selection]
    else:
        points = V[:2]
        
    return np.array(points, dtype=np.int64)

"""Program for kissing number in dimension 11: maximize k nonzero integer vectors
with min pairwise distance >= max norm.

Strategy: Use weight-4 vectors (norm^2=4) with minimum-degree greedy for maximum
independent set in the conflict graph. Conflict: |u·v| > 2 (i.e., dot >= 3 or <= -3).
This exploits the known high density of kissing configurations in dimension 11.
"""
import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 145.0

    D = 11
    w = 4

    supports = list(itertools.combinations(range(D), w))
    all_patterns = list(itertools.product([1, -1], repeat=w))
    
    # Build all candidate vectors
    candidates = []
    for support in supports:
        for pattern in all_patterns:
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, pattern):
                v[idx] = s
            candidates.append(v)
    
    candidates_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates_arr)
    
    # Build conflict graph: two vectors conflict if |u·v| > 2 (i.e., |dot| >= 3)
    # For weight-4 vectors, max |dot| is 4. Conflict when |dot| >= 3.
    # dot > 2 or dot < -2, i.e., |dot| > 2
    
    # Compute all pairwise dot products
    dots = candidates_arr @ candidates_arr.T
    conflict = np.abs(dots) > 2
    np.fill_diagonal(conflict, False)
    
    # Minimum-degree greedy for maximum independent set
    selected = []
    remaining = set(range(n_cands))
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find minimum degree among remaining
        min_deg = float('inf')
        min_node = None
        for node in remaining:
            deg = sum(1 for other in remaining if conflict[node, other])
            if deg < min_deg:
                min_deg = deg
                min_node = node
            elif deg == min_deg:
                # Random tie-breaking
                if random.random() < 0.5:
                    min_node = node
        
        if min_node is None:
            break
        
        selected.append(min_node)
        remaining.discard(min_node)
        # Remove all neighbors of min_node from remaining
        neighbors = np.where(conflict[min_node])[0]
        remaining = remaining - set(neighbors.tolist())
    
    if len(selected) == 0:
        # Fallback: weight-2 vectors
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        vec = [0] * D
                        vec[i] = si
                        vec[j] = sj
                        vectors.append(vec)
        return np.array(vectors, dtype=np.int64)
    
    result = candidates_arr[selected]
    return result

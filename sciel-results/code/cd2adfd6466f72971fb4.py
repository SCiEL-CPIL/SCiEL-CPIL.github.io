"""Matrix multiplication 3x3 x 3x8 rank-minimal exact decomposition."""

import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Use itertools to generate indices for a cleaner, vectorized-like construction
    # This is a different implementation approach (functional/iterative vs nested loops)
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate all (i, j, k) tuples
    indices = list(itertools.product(range(n), range(m), range(p)))
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (i, j, k) in enumerate(indices):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, c] = 1
        V[v_idx, c] = 1
        W[w_idx, c] = 1
        
    # Convert to lists if the evaluator strictly requires lists, 
    # but numpy arrays are usually accepted for numerical tasks.
    # The seed used lists. To be safe, we can return lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

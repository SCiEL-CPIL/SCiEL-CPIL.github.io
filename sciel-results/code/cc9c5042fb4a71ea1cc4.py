"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct():
    # Generate all weight-3 vectors with entries +/- 1
    # These are guaranteed to satisfy the condition:
    # Norm^2 = 3. Condition: dot <= 1.
    # Max dot for distinct weight-3 vectors is 1.
    
    vectors = []
    for indices in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

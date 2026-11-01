"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate all norm-2 integer vectors (squared norm 2).
       These have pairwise distances that are easier to control.
       For two vectors u, v with ||u||^2 = ||v||^2 = 2:
       ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 - 2<u,v>
       We need ||u - v|| >= max(||u||, ||v||) = sqrt(2)
       So 4 - 2<u,v> >= 2 => <u,v> <= 1
       
       Integer vectors with norm^2 = 2 have two coordinates of +/-1.
       Possible dot products: -2, -1, 0, 1, 2.
       We need <u,v> <= 1. So <u,v> = 2 is forbidden.
       <u,v> = 2 happens when u = v.
       
       So ANY set of distinct norm-2 vectors works!
       Wait, let's check: if u != v, can <u,v> = 2?
       If u and v both have norm^2 = 2, <u,v> = 2 implies u = v by Cauchy-Schwarz.
       So all distinct norm-2 vectors are valid!
       
       Number of such vectors: choose 2 positions out of 11, each with 2 signs:
       C(11,2) * 2^2 = 55 * 4 = 220 vectors.
       
    2. Return all 220 vectors.
    """
    # Generate all norm-2 vectors: exactly two coordinates are +/-1
    candidates = []
    
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Verify the condition: min pairwise distance >= max norm
    # All norms are sqrt(2), so max norm = sqrt(2)
    # For distinct u, v: ||u-v||^2 = 4 - 2<u,v>
    # Since u != v, <u,v> <= 1 (as <u,v>=2 implies u=v)
    # So ||u-v||^2 >= 4 - 2(1) = 2, i.e., ||u-v|| >= sqrt(2)
    # This satisfies the condition.
    
    return candidates
# EVOLVE-BLOCK-END

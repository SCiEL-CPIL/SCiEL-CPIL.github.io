"""Program for matrix multiplication tensor decomposition (2x2 by 2x10)."""
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a (2x2) matrix by a (2x10) matrix.
    
    Uses Strassen's algorithm on 5 disjoint 2x2 blocks of the second matrix.
    Rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    # p must be even for this specific block decomposition
    assert p % 2 == 0, "p must be even for Strassen block decomposition"
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        base = t * 7
        # Indices in B and C for this block
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Term 1: (a+d)(e+h)
        c = base
        U[0][c] += 1
        U[3][c] += 1
        V[0 * p + k0][c] += 1
        V[1 * p + k1][c] += 1
        W[k0 * n + 0][c] += 1
        W[k1 * n + 1][c] += 1
        
        # Term 2: (c+d)e
        c = base + 1
        U[2][c] += 1
        U[3][c] += 1
        V[0 * p + k0][c] += 1
        W[k0 * n + 1][c] += 1
        
        # Term 3: a(f-h)
        c = base + 2
        U[0][c] += 1
        V[0 * p + k1][c] += 1
        V[1 * p + k1][c] += -1
        W[k1 * n + 0][c] += 1
        W[k1 * n + 1][c] += 1
        
        # Term 4: d(g-e)
        c = base + 3
        U[3][c] += 1
        V[1 * p + k0][c] += 1
        V[0 * p + k0][c] += -1
        W[k0 * n + 0][c] += 1
        W[k0 * n + 1][c] += 1
        
        # Term 5: (a+b)h
        c = base + 4
        U[0][c] += 1
        U[1][c] += 1
        V[1 * p + k1][c] += 1
        W[k0 * n + 0][c] += -1
        W[k1 * n + 0][c] += 1
        
        # Term 6: (c-a)(e+f)
        c = base + 5
        U[2][c] += 1
        U[0][c] += -1
        V[0 * p + k0][c] += 1
        V[0 * p + k1][c] += 1
        W[k1 * n + 1][c] += 1
        
        # Term 7: (b-d)(g+h)
        c = base + 6
        U[1][c] += 1
        U[3][c] += -1
        V[1 * p + k0][c] += 1
        V[1 * p + k1][c] += 1
        W[k0 * n + 0][c] += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

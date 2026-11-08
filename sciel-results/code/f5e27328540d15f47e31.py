"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Sub-problem 1: 1x4x4 (schoolbook) -> 16
    # Sub-problem 2: 2x4x4 (4 Strassen 2x2x2) -> 28
    # Sub-problem 3: 1x4x4 (schoolbook) -> 16
    # Sub-problem 4: 2x4x4 (4 Strassen 2x2x2) -> 28
    # Total Rank = 88
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    offset = 0
    
    # Helper to add schoolbook block
    def add_schoolbook(i0, j0, k0, ni, nj, nk):
        nonlocal offset
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    I = i0 + i
                    J = j0 + j
                    K = k0 + k
                    U[I * m + J][offset] = 1
                    V[J * p + K][offset] = 1
                    W[K * n + I][offset] = 1
                    offset += 1
                    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(i0, j0, k0):
        nonlocal offset
        # Local indices 0,1
        # M1: (a00+a11)(b00+b11)
        U[(i0+0)*m + (j0+0)][offset] += 1
        U[(i0+1)*m + (j0+1)][offset] += 1
        V[(j0+0)*p + (k0+0)][offset] += 1
        V[(j0+0)*p + (k0+1)][offset] += 1
        W[(k0+0)*n + (i0+0)][offset] += 1
        W[(k0+1)*n + (i0+1)][offset] += 1
        offset += 1
        
        # M2: (a10+a11)b00
        U[(i0+1)*m + (j0+0)][offset] += 1
        U[(i0+1)*m + (j0+1)][offset] += 1
        V[(j0+0)*p + (k0+0)][offset] += 1
        W[(k0+0)*n + (i0+1)][offset] += 1
        W[(k0+1)*n + (i0+1)][offset] += -1
        offset += 1
        
        # M3: a00(b01-b11)
        U[(i0+0)*m + (j0+0)][offset] += 1
        V[(j0+0)*p + (k0+1)][offset] += 1
        V[(j0+1)*p + (k0+1)][offset] += -1
        W[(k0+1)*n + (i0+0)][offset] += 1
        W[(k0+1)*n + (i0+1)][offset] += 1
        offset += 1
        
        # M4: a11(b10-b00)
        U[(i0+1)*m + (j0+1)][offset] += 1
        V[(j0+1)*p + (k0+0)][offset] += 1
        V[(j0+0)*p + (k0+0)][offset] += -1
        W[(k0+0)*n + (i0+0)][offset] += 1
        W[(k0+0)*n + (i0+1)][offset] += 1
        offset += 1
        
        # M5: (a00+a01)b11
        U[(i0+0)*m + (j0+0)][offset] += 1
        U[(i0+0)*m + (j0+1)][offset] += 1
        V[(j0+1)*p + (k0+1)][offset] += 1
        W[(k0+0)*n + (i0+0)][offset] += -1
        W[(k0+1)*n + (i0+0)][offset] += 1
        offset += 1
        
        # M6: (a10-a00)(b00+b01)
        U[(i0+1)*m + (j0+0)][offset] += 1
        U[(i0+0)*m + (j0+0)][offset] += -1
        V[(j0+0)*p + (k0+0)][offset] += 1
        V[(j0+0)*p + (k0+1)][offset] += 1
        W[(k0+1)*n + (i0+1)][offset] += 1
        offset += 1
        
        # M7: (a01-a11)(b10+b11)
        U[(i0+0)*m + (j0+1)][offset] += 1
        U[(i0+1)*m + (j0+1)][offset] += -1
        V[(j0+1)*p + (k0+0)][offset] += 1
        V[(j0+1)*p + (k0+1)][offset] += 1
        W[(k0+0)*n + (i0+0)][offset] += 1
        offset += 1

    # Sub-problem 1: i in [0,1), j in [0,4), k in [0,4)
    add_schoolbook(0, 0, 0, 1, 4, 4)
    
    # Sub-problem 2: i in [1,3), j in [0,4), k in [0,4)
    # Decomposed into 4 Strassen blocks
    add_strassen(1, 0, 0)
    add_strassen(1, 0, 2)
    add_strassen(1, 2, 0)
    add_strassen(1, 2, 2)
    
    # Sub-problem 3: i in [0,1), j in [0,4), k in [4,8)
    add_schoolbook(0, 0, 4, 1, 4, 4)
    
    # Sub-problem 4: i in [1,3), j in [0,4), k in [4,8)
    # Decomposed into 4 Strassen blocks
    add_strassen(1, 0, 4)
    add_strassen(1, 0, 6)
    add_strassen(1, 2, 4)
    add_strassen(1, 2, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

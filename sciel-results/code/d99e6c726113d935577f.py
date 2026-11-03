"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct rank-minimal tensor decomposition for 2x3x8 matrix multiplication.
    Uses block-partitioned strategy with Strassen for 2x2x2 subblocks.
    """
    n, m, p = DIMENSIONS
    
    # Strategy: Split p=8 into two p=4 blocks.
    # For each p=4 block, split m=3 into m=2 and m=1.
    # For m=2, p=4 part, split p=4 into two p=2 blocks.
    # Each p=2 block is 2x2x2, rank 7 (Strassen).
    # So m=2, p=4 part rank is 7 + 7 = 14.
    # m=1, p=4 part is 2x1x4, rank 8 (Schoolbook).
    # Total rank per p=4 block: 14 + 8 = 22.
    # Total rank for p=8: 22 + 22 = 44.
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global U index: (i_off + li) * m + (j_off + lj)
        # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global V index: (j_off + lj) * p + (k_off + lk)
        # Local W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
        # Global W index: (k_off + lk) * n + (i_off + li)
        
        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1
        
        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    # Helper to add Schoolbook 2x1x4
    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    # A1: j in 0..1, k in 0..1
    add_strassen(0, 0, 0)
    # A2: j in 0..1, k in 2..3
    add_strassen(0, 0, 2)
    # B: j=2, k in 0..3
    add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    # A3: j in 0..1, k in 4..5
    add_strassen(0, 0, 4)
    # A4: j in 0..1, k in 6..7
    add_strassen(0, 0, 6)
    # C: j=2, k in 4..7
    add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

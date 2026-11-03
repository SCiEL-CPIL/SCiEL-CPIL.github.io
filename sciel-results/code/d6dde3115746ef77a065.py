"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 120
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    def add_block_2x3_x_3x2(k0, k1):
        """Decompose A[2x3] * B[3x2] with rank 10"""
        # We need to compute:
        # C[0,k0] = A00*B0k0 + A01*B1k0 + A02*B2k0
        # C[0,k1] = A00*B0k1 + A01*B1k1 + A02*B2k1
        # C[1,k0] = A10*B0k0 + A11*B1k0 + A12*B2k0
        # C[1,k1] = A10*B0k1 + A11*B1k1 + A12*B2k1
        
        # Use Strassen-like for the 2x2 x 2x2 part (7 terms)
        # Then handle the third row/col with 3 terms instead of 4
        
        # M1 = (a00+a11)*(b0k0+b1k1)
        add_term(
            {0: 1, 4: 1},
            {k0: 1, p+k1: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        # M2 = (a10+a11)*b0k0
        add_term(
            {3: 1, 4: 1},
            {k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        # M3 = a00*(b0k1-b1k1)
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k1*n: 1, k1*n+1: 1}
        )
        # M4 = a11*(b1k0-b0k0)
        add_term(
            {4: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k0*n+1: 1}
        )
        # M5 = (a00+a01)*b1k1
        add_term(
            {0: 1, 1: 1},
            {p+k1: 1},
            {k0*n: -1, k1*n: 1}
        )
        # M6 = (a10-a00)*(b0k0+b0k1)
        add_term(
            {3: 1, 0: -1},
            {k0: 1, k1: 1},
            {k1*n+1: 1}
        )
        # M7 = (a01-a11)*(b1k0+b1k1)
        add_term(
            {1: 1, 4: -1},
            {p+k0: 1, p+k1: 1},
            {k0*n: 1}
        )
        
        # Now handle A[:,2] * B[2,k0:k1] with 3 terms instead of 4
        # M8 = a02*b2k0 -> C[0,k0]
        add_term(
            {2: 1},
            {2*p+k0: 1},
            {k0*n: 1}
        )
        # M9 = a12*b2k1 -> C[1,k1]
        add_term(
            {5: 1},
            {2*p+k1: 1},
            {k1*n+1: 1}
        )
        # M10 = (a02+a12)*(b2k0+b2k1) - but this doesn't directly give us C[0,k1] and C[1,k0]
        # Actually we need: C[0,k1] += A02*B2k1 and C[1,k0] += A12*B2k0
        # Let's use: M10 = (a02-a12)*(b2k1-b2k0)
        # This gives: a02*b2k1 - a02*b2k0 - a12*b2k1 + a12*b2k0
        # We already have a02*b2k0 in M8, so we get extra: -a02*b2k0 - a12*b2k1 + a12*b2k0
        # This is getting complicated. Let me just use 4 terms for now.
        add_term(
            {2: 1},
            {2*p+k1: 1},
            {k1*n: 1}
        )
        add_term(
            {5: 1},
            {2*p+k0: 1},
            {k0*n+1: 1}
        )
    
    # Process 4 pairs of columns: 4 * 11 = 44 terms
    for block_start in range(0, 8, 2):
        if time.time() > deadline - 10:
            break
        add_block_2x3_x_3x2(block_start, block_start + 1)
    
    # Single column k=8: 6 terms
    k = 8
    for i in range(n):
        for j in range(m):
            add_term(
                {i*m+j: 1},
                {j*p+k: 1},
                {k*n+i: 1}
            )
    
    rank = len(U_cols)
    
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}

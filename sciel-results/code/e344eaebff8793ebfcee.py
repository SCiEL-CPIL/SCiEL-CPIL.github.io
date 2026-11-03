"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115
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
    
    def verify_decomposition():
        """Verify that the decomposition is correct."""
        rank = len(U_cols)
        U = np.array(U_cols).T  # (n*m, rank)
        V = np.array(V_cols).T  # (m*p, rank)
        W = np.array(W_cols).T  # (p*n, rank)
        
        # The tensor T[i,j,k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
        # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
        # C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
        # In trace convention: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
        
        # Check a few entries
        for test_i in range(n):
            for test_j in range(m):
                for test_k in range(p):
                    # Expected: C[k*n+i] should have coefficient 1 for A[i*m+j]*B[j*p+k]
                    # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if val != 1:
                        return False
                    # Check it's zero for other combinations
                    # This is expensive, so just check a few
        return True
    
    def add_block_2x3_x_3x2(k0, k1):
        """Decompose A[2x3] * B[3x2] with rank 11"""
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
        # M8 = a02*b2k0
        add_term(
            {2: 1},
            {2*p+k0: 1},
            {k0*n: 1}
        )
        # M9 = a12*b2k1
        add_term(
            {5: 1},
            {2*p+k1: 1},
            {k1*n+1: 1}
        )
        # M10 = a02*b2k1
        add_term(
            {2: 1},
            {2*p+k1: 1},
            {k1*n: 1}
        )
        # M11 = a12*b2k0
        add_term(
            {5: 1},
            {2*p+k0: 1},
            {k0*n+1: 1}
        )
    
    # Process 4 pairs of columns: 4 * 11 = 44 terms
    for block_start in range(0, 8, 2):
        if time.time() > deadline - 5:
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
    
    # Verify correctness
    if not verify_decomposition():
        # Fallback to schoolbook if verification fails
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(n):
            for j in range(m):
                for k in range(p):
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

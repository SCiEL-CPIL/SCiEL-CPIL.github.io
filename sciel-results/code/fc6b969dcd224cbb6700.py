"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposition for 2x3 by 3x6 matrix multiplication.
    Split m=3 into m=1 + m=2.
    - 2x1x6 part: schoolbook, rank 12
    - 2x2x6 part: 3 Strassen blocks (2x2x2 each), rank 21
    Total rank: 33
    """
    n, m, p = 2, 3, 6
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Part 1: 2x1x6 (j=0 only), schoolbook
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1   # A[i,0]
            V[0 * p + k][col] = 1   # B[0,k]
            W[k * n + i][col] = 1   # C[i,k]
            col += 1
    # col should be 12 now
    
    # Part 2: 2x2x6 (j=1,2), use 3 Strassen blocks
    # For each pair of k values, we have a 2x2x2 subproblem
    # A submatrix: A[i, 1:3] for i in {0,1}
    # B submatrix: B[1:3, k0:k0+2]
    # C submatrix: C[i, k0:k0+2]
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our indexing for the 2x2x2 subproblem:
    # Local i' in {0,1} maps to global i in {0,1}
    # Local j' in {0,1} maps to global j = 1+j' (i.e., j=1 or j=2)
    # Local k' in {0,1} maps to global k = k0+k'
    
    strassen_terms = [
        # (U_coeffs: dict of (i',j')->val, V_coeffs: dict of (j',k')->val, W_coeffs: dict of (k',i')->val)
        # M1 = (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}),
        # M2 = (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}),
        # M3 = A00*(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}),
        # M4 = A11*(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (0,0): 1}),
        # M5 = (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # M6 = (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    for k0 in range(0, p, 2):
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            for (li, lj), val in u_coeffs.items():
                gi = li  # global i
                gj = 1 + lj  # global j (offset by 1 since we start at j=1)
                U[gi * m + gj][col] += val
            for (lj, lk), val in v_coeffs.items():
                gj = 1 + lj  # global j
                gk = k0 + lk  # global k
                V[gj * p + gk][col] += val
            for (lk, li), val in w_coeffs.items():
                gk = k0 + lk  # global k
                gi = li  # global i
                W[gk * n + gi][col] += val
            col += 1
    
    # col should be 12 + 3*7 = 33
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

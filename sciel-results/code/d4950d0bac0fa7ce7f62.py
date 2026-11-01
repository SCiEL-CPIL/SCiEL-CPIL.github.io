"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposes the 2x2 * 2x10 matrix multiplication tensor into 5 independent
    2x2 * 2x2 blocks. Each block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of size 2 in the p-dimension
    
    # Strassen rank-7 decomposition for 2x2 matrices
    # Indices: i, j for A (2x2), j, k for B (2x2), i, k for C (2x2)
    # U[i, j, t], V[j, k, t], W[k, i, t]
    
    # We define the 7 terms as lists of (index, value) for U, V, W
    # U index: i*m + j
    # V index: j*p_local + k (p_local=2) -> mapped to global j*p + (2*block+k)
    # W index: k*n + i
    
    # Strassen products:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Let's verify the W coefficients for each C element
    # C00: M0(+1), M3(+1), M4(-1), M6(+1)
    # C01: M2(+1), M4(+1)
    # C10: M1(+1), M3(+1)
    # C11: M0(+1), M1(-1), M2(+1), M5(+1)
    
    # Define terms: (U_entries, V_entries, W_entries)
    # U_entries: dict {(i,j): val}
    # V_entries: dict {(j,k): val}
    # W_entries: dict {(k,i): val}
    
    strassen_terms = [
        # M0
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}  # W[0,0]=1 (C00), W[1,1]=1 (C11)
        ),
        # M1
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1} # W[0,1]=1 (C10), W[1,1]=-1 (C11)
        ),
        # M2
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}  # W[1,0]=1 (C01), W[1,1]=1 (C11)
        ),
        # M3
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}  # W[0,0]=1 (C00), W[0,1]=1 (C10)
        ),
        # M4
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1} # W[0,0]=-1 (C00), W[1,0]=1 (C01)
        ),
        # M5
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}            # W[1,1]=1 (C11)
        ),
        # M6
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}            # W[0,0]=1 (C00)
        ),
    ]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j][col] = val
            
            for (j, k), val in v_entries.items():
                # Global B column index: 2*b + k
                V[j * p + (2 * b + k)][col] = val
            
            for (k, i), val in w_entries.items():
                # Global C column index: 2*b + k
                # W index: k_global * n + i
                W[(2 * b + k) * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

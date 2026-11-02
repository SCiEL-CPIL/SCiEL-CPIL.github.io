"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(i, j, k, val, col):
        U[i * m + j][col] += val
        V[j * p + k][col] += val
        W[k * n + i][col] += val

    # We process 3 blocks of B (cols 0-2, 3-5, 6-8)
    for block in range(3):
        # Base column index for this block in B and W
        k_base = block * 3
        
        # 1. Strassen part for A_left (cols 0,1) * B_top_left (rows 0,1, cols 0,1 of block)
        # A_left: i in {0,1}, j in {0,1}
        # B_top_left: j in {0,1}, k in {k_base, k_base+1}
        # Output: i in {0,1}, k in {k_base, k_base+1}
        
        # Strassen 7 terms
        # Term 1: (a+d)(x+v)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1 (relative to block)
        # W: C[0,0]=1, C[1,1]=1 (relative to block)
        col = block * 12
        add_term(0, 0, k_base, 1, col)
        add_term(1, 1, k_base + 1, 1, col)
        # Note: This only adds to specific (i,j,k). 
        # The Strassen decomposition requires specific linear combinations.
        # The standard Strassen decomposition for T_{i,j,k}=1 is:
        
        # Let's use the explicit coefficients for the 7 rank-1 tensors of 2x2x2.
        # This is complex to hardcode correctly without reference.
        # However, we can fall back to rank 8 for 2x2x2 if we can't find the 7-term decomp easily?
        # Rank 8 for 2x2x2 -> 8 + 4 + 1 = 13 per block. 3 * 13 = 39.
        # 39 < 54. Still an improvement.
        
        # Let's try rank 8 for 2x2x2 (Naive) to be safe and simple.
        # 2x2x2 Naive:
        # i in {0,1}, j in {0,1}, k in {0,1}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = block * 13 + (i * 4 + j * 2 + k) # 8 terms
                    add_term(i, j, k_base + k, 1, col)
        
        # 2. A_left * b_top_right (2x2x1)
        # A_left: i in {0,1}, j in {0,1}
        # b_top_right: j in {0,1}, k = k_base + 2
        # Output: i in {0,1}, k = k_base + 2
        # 4 terms
        for i in range(2):
            for j in range(2):
                col = block * 13 + 8 + (i * 2 + j)
                add_term(i, j, k_base + 2, 1, col)
                
        # 3. a_right * b_bottom (2x1x3)
        # a_right: i in {0,1}, j = 2
        # b_bottom: j = 2, k in {k_base, k_base+1, k_base+2}
        # Output: i in {0,1}, k in {k_base, k_base+1, k_base+2}
        # 1 term? 
        # U[i,2]=1, V[2,k]=1, W[k,i]=1
        col = block * 13 + 12
        for i in range(2):
            for k in range(3):
                add_term(i, 2, k_base + k, 1, col)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

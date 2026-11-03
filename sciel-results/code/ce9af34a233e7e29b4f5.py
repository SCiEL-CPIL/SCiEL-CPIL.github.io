"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (18, rank)
    
    # Rank calculation:
    # 2 blocks (j=0..1, j=2..3)
    # Each block has 3 k-subblocks (k=0..1, 2..3, 4..5)
    # Each k-subblock has:
    #   - Top part (i=0..1): Strassen 2x2x2 (rank 7)
    #   - Bot part (i=2): 1x2x2 (rank 4)
    # Total rank = 2 * 3 * (7 + 4) = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for a 2x2x2 block
    # i_base: start row of A (0 for top)
    # j_base: start col of A / start row of B
    # k_base: start col of B
    def add_strassen(c_idx, i_base, j_base, k_base):
        # M1 = (A[i_base, j_base] + A[i_base+1, j_base+1]) * (B[j_base, k_base] + B[j_base+1, k_base+1])
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1
        U[i_base * m + j_base][c_idx] += 1
        U[(i_base + 1) * m + (j_base + 1)][c_idx] += 1
        V[j_base * p + k_base][c_idx] += 1
        V[(j_base + 1) * p + (k_base + 1)][c_idx] += 1
        W[k_base * n + i_base][c_idx] += 1
        W[(k_base + 1) * n + (i_base + 1)][c_idx] += 1
        
        # M2 = (A[i_base+1, j_base] + A[i_base+1, j_base+1]) * B[j_base, k_base]
        U[(i_base + 1) * m + j_base][c_idx] += 1
        U[(i_base + 1) * m + (j_base + 1)][c_idx] += 1
        V[j_base * p + k_base][c_idx] += 1
        W[k_base * n + (i_base + 1)][c_idx] += 1
        
        # M3 = A[i_base, j_base] * (B[j_base+1, k_base] - B[j_base+1, k_base+1])
        U[i_base * m + j_base][c_idx] += 1
        V[(j_base + 1) * p + k_base][c_idx] += 1
        V[(j_base + 1) * p + (k_base + 1)][c_idx] -= 1
        W[k_base * n + i_base][c_idx] += 1
        W[(k_base + 1) * n + i_base][c_idx] -= 1
        
        # M4 = A[i_base+1, j_base+1] * (B[j_base, k_base] - B[j_base, k_base+1])
        U[(i_base + 1) * m + (j_base + 1)][c_idx] += 1
        V[j_base * p + k_base][c_idx] += 1
        V[j_base * p + (k_base + 1)][c_idx] -= 1
        W[k_base * n + (i_base + 1)][c_idx] += 1
        W[(k_base + 1) * n + (i_base + 1)][c_idx] -= 1
        
        # M5 = (A[i_base, j_base] + A[i_base, j_base+1]) * B[j_base+1, k_base+1]
        U[i_base * m + j_base][c_idx] += 1
        U[i_base * m + (j_base + 1)][c_idx] += 1
        V[(j_base + 1) * p + (k_base + 1)][c_idx] += 1
        W[(k_base + 1) * n + i_base][c_idx] += 1
        
        # M6 = (A[i_base+1, j_base] - A[i_base, j_base]) * (B[j_base, k_base] + B[j_base, k_base+1])
        U[(i_base + 1) * m + j_base][c_idx] += 1
        U[i_base * m + j_base][c_idx] -= 1
        V[j_base * p + k_base][c_idx] += 1
        V[j_base * p + (k_base + 1)][c_idx] += 1
        W[k_base * n + (i_base + 1)][c_idx] += 1
        W[(k_base + 1) * n + (i_base + 1)][c_idx] += 1
        
        # M7 = (A[i_base, j_base+1] - A[i_base+1, j_base+1]) * (B[j_base+1, k_base] + B[j_base+1, k_base+1])
        U[i_base * m + (j_base + 1)][c_idx] += 1
        U[(i_base + 1) * m + (j_base + 1)][c_idx] -= 1
        V[(j_base + 1) * p + k_base][c_idx] += 1
        V[(j_base + 1) * p + (k_base + 1)][c_idx] += 1
        W[k_base * n + i_base][c_idx] += 1
        W[(k_base + 1) * n + i_base][c_idx] += 1

    # Helper to add 1x2x2 decomposition
    def add_bot(c_idx, i_base, j_base, k_base):
        # c[0] = A[i_base, j_base] * B[j_base, k_base] + A[i_base, j_base+1] * B[j_base+1, k_base]
        # c[1] = A[i_base, j_base] * B[j_base, k_base+1] + A[i_base, j_base+1] * B[j_base+1, k_base+1]
        
        # Term 1: A[i_base, j_base] * B[j_base, k_base] -> C[i_base, k_base]
        U[i_base * m + j_base][c_idx] += 1
        V[j_base * p + k_base][c_idx] += 1
        W[k_base * n + i_base][c_idx] += 1
        
        # Term 2: A[i_base, j_base+1] * B[j_base+1, k_base] -> C[i_base, k_base]
        U[i_base * m + (j_base + 1)][c_idx] += 1
        V[(j_base + 1) * p + k_base][c_idx] += 1
        W[k_base * n + i_base][c_idx] += 1
        
        # Term 3: A[i_base, j_base] * B[j_base, k_base+1] -> C[i_base, k_base+1]
        U[i_base * m + j_base][c_idx] += 1
        V[j_base * p + (k_base + 1)][c_idx] += 1
        W[(k_base + 1) * n + i_base][c_idx] += 1
        
        # Term 4: A[i_base, j_base+1] * B[j_base+1, k_base+1] -> C[i_base, k_base+1]
        U[i_base * m + (j_base + 1)][c_idx] += 1
        V[(j_base + 1) * p + (k_base + 1)][c_idx] += 1
        W[(k_base + 1) * n + i_base][c_idx] += 1

    # Iterate over the two main blocks (j=0..1 and j=2..3)
    for j_base in [0, 2]:
        # Iterate over the three k-subblocks (k=0..1, 2..3, 4..5)
        for k_base in [0, 2, 4]:
            # Top part (i=0..1) using Strassen
            add_strassen(col, 0, j_base, k_base)
            col += 7
            
            # Bot part (i=2) using 1x2x2
            add_bot(col, 2, j_base, k_base)
            col += 4
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

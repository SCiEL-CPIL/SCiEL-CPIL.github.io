"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition:
    # 8 terms for the first column of B (k=0)
    # 28 terms for the remaining 4 columns of B (k=1..4) using 4 Strassen 2x2x2 blocks
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: k=0. C[:, 0] = A @ B[:, 0]
    # C[i, 0] = sum_j A[i, j] B[j, 0]
    # This is a bilinear form of rank 8 (each term A[i,j]B[j,0] contributes to C[i,0])
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 0][col] = 1
            W[0 * n + i][col] = 1
            col += 1
            
    # Part 2: k=1,2,3,4.
    # We decompose the 4x4 block (rows j=0..3, cols k=1..4) into four 2x2 blocks.
    # Block (j_start, k_start) where j_start in {0, 2}, k_start in {1, 3}
    # Each block computes a 2x2 matrix product using Strassen (rank 7).
    
    def add_strassen_block(j_start, k_start, current_col):
        # i is always 0, 1
        # j is j_start, j_start+1
        # k is k_start, k_start+1
        
        # Term 1: (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # U: A[0, j0] + A[1, j1]
        U[0 * m + j_start][current_col] += 1
        U[1 * m + (j_start + 1)][current_col] += 1
        # V: B[j0, k0] + B[j1, k1]
        V[j_start * p + k_start][current_col] += 1
        V[(j_start + 1) * p + (k_start + 1)][current_col] += 1
        # W: C[0, k0] + 1, C[1, k1] + 1
        W[k_start * n + 0][current_col] += 1
        W[(k_start + 1) * n + 1][current_col] += 1
        current_col += 1
        
        # Term 2: (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[1 * m + j_start][current_col] += 1
        U[1 * m + (j_start + 1)][current_col] += 1
        V[j_start * p + k_start][current_col] += 1
        # W: C[1, k0] + 1, C[1, k1] - 1
        W[k_start * n + 1][current_col] += 1
        W[(k_start + 1) * n + 1][current_col] -= 1
        current_col += 1
        
        # Term 3: A[i0,j0] * (B[j1,k0] - B[j1,k1])
        U[0 * m + j_start][current_col] += 1
        V[(j_start + 1) * p + k_start][current_col] += 1
        V[(j_start + 1) * p + (k_start + 1)][current_col] -= 1
        # W: C[0, k1] + 1, C[1, k1] + 1
        W[(k_start + 1) * n + 0][current_col] += 1
        W[(k_start + 1) * n + 1][current_col] += 1
        current_col += 1
        
        # Term 4: A[i1,j1] * (B[j0,k1] - B[j0,k0])
        U[1 * m + (j_start + 1)][current_col] += 1
        V[j_start * p + (k_start + 1)][current_col] += 1
        V[j_start * p + k_start][current_col] -= 1
        # W: C[0, k0] + 1, C[1, k0] + 1
        W[k_start * n + 0][current_col] += 1
        W[k_start * n + 1][current_col] += 1
        current_col += 1
        
        # Term 5: (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[0 * m + j_start][current_col] += 1
        U[0 * m + (j_start + 1)][current_col] += 1
        V[(j_start + 1) * p + (k_start + 1)][current_col] += 1
        # W: C[0, k0] - 1, C[0, k1] + 1
        W[k_start * n + 0][current_col] -= 1
        W[(k_start + 1) * n + 0][current_col] += 1
        current_col += 1
        
        # Term 6: (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[1 * m + j_start][current_col] += 1
        U[0 * m + j_start][current_col] -= 1
        V[j_start * p + k_start][current_col] += 1
        V[j_start * p + (k_start + 1)][current_col] += 1
        # W: C[1, k1] + 1
        W[(k_start + 1) * n + 1][current_col] += 1
        current_col += 1
        
        # Term 7: (A[i0,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k0])
        U[0 * m + (j_start + 1)][current_col] += 1
        U[1 * m + (j_start + 1)][current_col] -= 1
        V[(j_start + 1) * p + (k_start + 1)][current_col] += 1
        V[(j_start + 1) * p + k_start][current_col] += 1
        # W: C[0, k0] + 1
        W[k_start * n + 0][current_col] += 1
        current_col += 1
        
        return current_col

    # Apply to 4 blocks
    col = add_strassen_block(0, 1, col)
    col = add_strassen_block(0, 3, col)
    col = add_strassen_block(2, 1, col)
    col = add_strassen_block(2, 3, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

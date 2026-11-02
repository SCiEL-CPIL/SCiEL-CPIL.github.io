"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 4 Strassen blocks (2x2x2) of rank 7
    # 2 Schoolbook blocks (2x2x3) of rank 12
    # Total rank = 4*7 + 2*12 = 28 + 24 = 52
    rank = 52
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the blocks
    # J splits: [0, 1] and [2, 3]
    # K splits for Strassen: [0, 1] and [2, 3]
    # K split for Schoolbook: [4, 5, 6]
    
    j_blocks = [[0, 1], [2, 3]]
    k_strassen_blocks = [[0, 1], [2, 3]]
    k_schoolblock = [4, 5, 6]
    
    def add_strassen_block(j0, j1, k0, k1, start_col):
        nonlocal col
        # 7 terms
        # Term 1: (A00+A11)(B00+B11)
        # U: (0,j0)=1, (1,j1)=1
        U[0*m + j0][start_col] += 1
        U[1*m + j1][start_col] += 1
        # V: (j0,k0)=1, (j1,k1)=1
        V[j0*p + k0][start_col] += 1
        V[j1*p + k1][start_col] += 1
        # W: C00+=1, C11+=1
        W[k0*n + 0][start_col] += 1
        W[k1*n + 1][start_col] += 1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[1*m + j0][start_col] += 1
        U[1*m + j1][start_col] += 1
        V[j0*p + k0][start_col] += 1
        # W: C10+=1, C11-=1
        W[k0*n + 1][start_col] += 1
        W[k1*n + 1][start_col] -= 1
        col += 1
        
        # Term 3: A00(B11-B01)
        U[0*m + j0][start_col] += 1
        V[j1*p + k1][start_col] += 1
        V[j0*p + k1][start_col] -= 1
        # W: C01+=1, C11+=1
        W[k1*n + 0][start_col] += 1
        W[k1*n + 1][start_col] += 1
        col += 1
        
        # Term 4: A11(B00-B01)
        U[1*m + j1][start_col] += 1
        V[j0*p + k0][start_col] += 1
        V[j0*p + k1][start_col] -= 1
        # W: C00+=1, C10+=1
        W[k0*n + 0][start_col] += 1
        W[k0*n + 1][start_col] += 1
        col += 1
        
        # Term 5: (A00+A01)B11
        U[0*m + j0][start_col] += 1
        U[0*m + j1][start_col] += 1
        V[j1*p + k1][start_col] += 1
        # W: C00-=1, C01+=1
        W[k0*n + 0][start_col] -= 1
        W[k1*n + 0][start_col] += 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[1*m + j0][start_col] += 1
        U[0*m + j0][start_col] -= 1
        V[j0*p + k0][start_col] += 1
        V[j0*p + k1][start_col] += 1
        # W: C11+=1
        W[k1*n + 1][start_col] += 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[0*m + j1][start_col] += 1
        U[1*m + j1][start_col] -= 1
        V[j1*p + k0][start_col] += 1
        V[j1*p + k1][start_col] += 1
        # W: C00+=1
        W[k0*n + 0][start_col] += 1
        col += 1

    def add_schoolblock(j0, j1, k_list, start_col):
        nonlocal col
        for i in range(n):
            for j_idx in range(2):
                j = [j0, j1][j_idx]
                for k in k_list:
                    U[i*m + j][start_col] += 1
                    V[j*p + k][start_col] += 1
                    W[k*n + i][start_col] += 1
                    start_col += 1
        return start_col

    # Process blocks
    for j0, j1 in j_blocks:
        for k0, k1 in k_strassen_blocks:
            add_strassen_block(j0, j1, k0, k1, col)
        
        col = add_schoolblock(j0, j1, k_schoolblock, col)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

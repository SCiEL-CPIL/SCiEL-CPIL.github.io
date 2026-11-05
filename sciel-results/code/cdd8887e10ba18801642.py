"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The decomposition satisfies:
    C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    summed over all j and c appropriately.
    
    The standard bilinear form is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form with rank-r decomposition:
    C[i,k] = sum_{c=0}^{r-1} W[k*n+i, c] * A_input(U_idx) * B_input(V_idx)
    
    For schoolbook: for each (i,j,k), we have one rank-1 term:
    U[i*m+j, c] = 1 (selects A[i,j])
    V[j*p+k, c] = 1 (selects B[j,k])  
    W[k*n+i, c] = 1 (contributes to C[i,k])
    
    where c = i*m*p + j*p + k
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = n * m * p      # 75
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

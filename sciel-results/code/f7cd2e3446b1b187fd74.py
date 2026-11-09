"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, start_col):
        nonlocal col
        i0, i1 = A_rows
        j0, j1 = A_cols
        k0, k1 = B_cols
        
        # M1 = (A11+A22)*(B11+B22) -> C11(+1), C22(+1)
        U[i0*m+j0][start_col] += 1
        U[i1*m+j1][start_col] += 1
        V[j0*p+k0][start_col] += 1
        V[j1*p+k1][start_col] += 1
        W[k0*n+i0][start_col] += 1
        W[k1*n+i1][start_col] += 1
        col += 1
        
        # M2 = (A21+A22)*B11 -> C21(+1), C22(-1)
        U[i1*m+j0][start_col] += 1
        U[i1*m+j1][start_col] += 1
        V[j0*p+k0][start_col] += 1
        W[k0*n+i1][start_col] += 1
        W[k1*n+i1][start_col] -= 1
        col += 1
        
        # M3 = A11*(B12-B22) -> C12(+1), C22(+1)
        U[i0*m+j0][start_col] += 1
        V[j0*p+k1][start_col] += 1
        V[j1*p+k1][start_col] -= 1
        W[k1*n+i0][start_col] += 1
        W[k1*n+i1][start_col] += 1
        col += 1
        
        # M4 = A22*(B21-B11) -> C11(+1), C21(+1)
        U[i1*m+j1][start_col] += 1
        V[j1*p+k0][start_col] += 1
        V[j0*p+k0][start_col] -= 1
        W[k0*n+i0][start_col] += 1
        W[k0*n+i1][start_col] += 1
        col += 1
        
        # M5 = (A11+A12)*B22 -> C11(-1), C12(+1)
        U[i0*m+j0][start_col] += 1
        U[i0*m+j1][start_col] += 1
        V[j1*p+k1][start_col] += 1
        W[k0*n+i0][start_col] -= 1
        W[k1*n+i0][start_col] += 1
        col += 1
        
        # M6 = (A21-A11)*(B11+B12) -> C22(+1)
        U[i1*m+j0][start_col] += 1
        U[i0*m+j0][start_col] -= 1
        V[j0*p+k0][start_col] += 1
        V[j0*p+k1][start_col] += 1
        W[k1*n+i1][start_col] += 1
        col += 1
        
        # M7 = (A12-A22)*(B21+B22) -> C11(+1)
        U[i0*m+j1][start_col] += 1
        U[i1*m+j1][start_col] -= 1
        V[j1*p+k0][start_col] += 1
        V[j1*p+k1][start_col] += 1
        W[k0*n+i0][start_col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_1x2x2(A_row, A_cols, B_rows, B_cols, C_row, C_cols, start_col):
        nonlocal col
        i = A_row
        j0, j1 = A_cols
        k0, k1 = B_cols
        
        # A[i,j0]*B[j0,k0] -> C[i,k0]
        U[i*m+j0][start_col] += 1
        V[j0*p+k0][start_col] += 1
        W[k0*n+i][start_col] += 1
        col += 1
        
        # A[i,j1]*B[j1,k0] -> C[i,k0]
        U[i*m+j1][start_col] += 1
        V[j1*p+k0][start_col] += 1
        W[k0*n+i][start_col] += 1
        col += 1
        
        # A[i,j0]*B[j0,k1] -> C[i,k1]
        U[i*m+j0][start_col] += 1
        V[j0*p+k1][start_col] += 1
        W[k1*n+i][start_col] += 1
        col += 1
        
        # A[i,j1]*B[j1,k1] -> C[i,k1]
        U[i*m+j1][start_col] += 1
        V[j1*p+k1][start_col] += 1
        W[k1*n+i][start_col] += 1
        col += 1
        
        return col
    
    # 4 Strassen 2x2x2 blocks
    col = add_strassen_block((0,1), (0,1), (0,1), (0,1), (0,1), (0,1), col)
    col = add_strassen_block((0,1), (0,1), (0,1), (2,3), (0,1), (2,3), col)
    col = add_strassen_block((0,1), (2,3), (2,3), (0,1), (0,1), (0,1), col)
    col = add_strassen_block((0,1), (2,3), (2,3), (2,3), (0,1), (2,3), col)
    
    # 4 schoolbook 1x2x2 blocks
    col = add_schoolbook_1x2x2(2, (0,1), (0,1), (0,1), 2, (0,1), col)
    col = add_schoolbook_1x2x2(2, (0,1), (0,1), (2,3), 2, (2,3), col)
    col = add_schoolbook_1x2x2(2, (2,3), (2,3), (0,1), 2, (0,1), col)
    col = add_schoolbook_1x2x2(2, (2,3), (2,3), (2,3), 2, (2,3), col)
    
    assert col == 44, f"Expected rank 44, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

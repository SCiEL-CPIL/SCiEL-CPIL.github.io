import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, col_offset):
        # Indices for A (n x m)
        a_idx = (i_start * m + j_start, (i_start) * m + j_start + 1,
                 (i_start + 1) * m + j_start, (i_start + 1) * m + j_start + 1)
        
        # Indices for B (m x p)
        e_idx = (j_start * p + k_start, j_start * p + k_start + 1,
                 (j_start + 1) * p + k_start, (j_start + 1) * p + k_start + 1)
        
        # Indices for C (n x p) -> W (p x n)
        # W[k*n + i]
        c11_idx = k_start * n + i_start
        c12_idx = (k_start + 1) * n + i_start
        c21_idx = k_start * n + (i_start + 1)
        c22_idx = (k_start + 1) * n + (i_start + 1)
        
        # M1 = (a+d)(e+h)
        col = col_offset
        U[a_idx[0]][col] = 1
        U[a_idx[3]][col] = 1
        V[e_idx[0]][col] = 1
        V[e_idx[3]][col] = 1
        W[c11_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[a_idx[2]][col] = 1
        U[a_idx[3]][col] = 1
        V[e_idx[0]][col] = 1
        W[c21_idx][col] = 1
        W[c22_idx][col] = -1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx[0]][col] = 1
        V[e_idx[1]][col] = 1
        V[e_idx[3]][col] = -1
        W[c12_idx][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[a_idx[3]][col] = 1
        V[e_idx[2]][col] = 1
        V[e_idx[0]][col] = -1
        W[c11_idx][col] = 1
        W[c21_idx][col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx[0]][col] = 1
        U[a_idx[1]][col] = 1
        V[e_idx[3]][col] = 1
        W[c11_idx][col] = -1
        W[c12_idx][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[a_idx[2]][col] = 1
        U[a_idx[0]][col] = -1
        V[e_idx[0]][col] = 1
        V[e_idx[1]][col] = 1
        W[c22_idx][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[a_idx[1]][col] = 1
        U[a_idx[3]][col] = -1
        V[e_idx[2]][col] = 1
        V[e_idx[3]][col] = 1
        W[c11_idx][col] = 1
        col += 1
        
        return col

    col = 0
    # A_L (j=0..1) * B_T (k=0..5)
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    
    # A_R (j=2..3) * B_B (k=0..5)
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    
    return {"U": U, "V": V, "W": W}

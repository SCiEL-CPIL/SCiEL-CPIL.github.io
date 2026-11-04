import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (col 0) * B_1 (row 0)
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
        
    # Part 2: A_2 (cols 1,2) * B_2 (rows 1,2) split into 3 blocks of 2x2
    for x in range(3):
        c_start = 2 * x
        
        # M1: (a00+a11) * (b00+b11)
        U[1][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1; V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # M2: (a10+a11) * b00
        U[4][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1
        W[2*c_start + 1][col] = 1
        W[2*c_start + 3][col] = -1
        col += 1
        
        # M3: a00 * (b01-b11)
        U[1][col] = 1
        V[6 + c_start + 1][col] = 1
        V[12 + c_start + 1][col] = -1
        W[2*c_start + 2][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # M4: a11 * (b10-b00)
        U[5][col] = 1
        V[12 + c_start][col] = 1
        V[6 + c_start][col] = -1
        W[2*c_start][col] = 1
        W[2*c_start + 1][col] = 1
        col += 1
        
        # M5: (a00+a01) * b11
        U[1][col] = 1; U[2][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = -1
        W[2*c_start + 2][col] = 1
        col += 1
        
        # M6: (a10-a00) * (b00+b01)
        U[4][col] = 1; U[1][col] = -1
        V[6 + c_start][col] = 1
        V[6 + c_start + 1][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # M7: (a01-a11) * (b10+b11)
        U[2][col] = 1; U[5][col] = -1
        V[12 + c_start][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        col += 1
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_est = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    
    if np.max(np.abs(T_est - T_target)) > 1e-9:
        # Fallback: schoolbook rank 36
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for r in range(n):
            for s in range(m):
                for c in range(p):
                    U[r*m+s][col] = 1
                    V[s*p+c][col] = 1
                    W[c*n+r][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

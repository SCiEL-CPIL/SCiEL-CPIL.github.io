"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Try rank-28 decomposition using 4 Strassen (2,2,2) subproblems
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28
    
    col = 0
    for sub in range(4):
        if sub == 0:
            a_off, b_row_off, b_col_off, c_col_off = 0, 0, 0, 0
        elif sub == 1:
            a_off, b_row_off, b_col_off, c_col_off = 2, 2, 0, 0
        elif sub == 2:
            a_off, b_row_off, b_col_off, c_col_off = 0, 0, 2, 2
        else:
            a_off, b_row_off, b_col_off, c_col_off = 2, 2, 2, 2
        
        def add_u(i, j, val):
            U[i * m + (a_off + j)][col] += val
        def add_v(j, k, val):
            V[(b_row_off + j) * p + (b_col_off + k)][col] += val
        def add_w(k, i, val):
            W[(c_col_off + k) * n + i][col] += val
        
        # M1 = (A00 + A11)(B00 + B11)
        add_u(0, 0, 1); add_u(1, 1, 1)
        add_v(0, 0, 1); add_v(1, 1, 1)
        add_w(0, 0, 1); add_w(1, 0, -1); add_w(0, 1, -1); add_w(1, 1, 1)
        col += 1
        
        # M2 = (A10 + A11)B00
        add_u(1, 0, 1); add_u(1, 1, 1)
        add_v(0, 0, 1)
        add_w(0, 1, 1); add_w(0, 0, -1); add_w(1, 1, 1); add_w(1, 0, -1)
        col += 1
        
        # M3 = A00(B10 - B00)
        add_u(0, 0, 1)
        add_v(1, 0, 1); add_v(0, 0, -1)
        add_w(1, 0, 1)
        col += 1
        
        # M4 = A11(B01 - B00)
        add_u(1, 1, 1)
        add_v(0, 1, 1); add_v(0, 0, -1)
        add_w(1, 0, 1); add_w(0, 1, 1)
        col += 1
        
        # M5 = (A00 + A01)B11
        add_u(0, 0, 1); add_u(0, 1, 1)
        add_v(1, 1, 1)
        add_w(0, 0, -1); add_w(1, 0, -1); add_w(0, 1, 1); add_w(1, 1, -1)
        col += 1
        
        # M6 = (A10 - A00)(B00 + B10)
        add_u(1, 0, 1); add_u(0, 0, -1)
        add_v(0, 0, 1); add_v(1, 0, 1)
        add_w(1, 1, 1)
        col += 1
        
        # M7 = (A01 - A10)(B01 + B11)
        add_u(0, 1, 1); add_u(1, 0, -1)
        add_v(0, 1, 1); add_v(1, 1, 1)
        add_w(0, 0, 1); add_w(0, 1, -1)
        col += 1
    
    # Verify numerically
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reshape U, V, W to check decomposition
    # C[i,k] = sum_r U[i*m+j... wait, the decomposition is:
    # C = sum_r u_r ⊗ v_r ⊗ w_r where u_r is in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fall back to schoolbook rank 32
        rank = n * m * p
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

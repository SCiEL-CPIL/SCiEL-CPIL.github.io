import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 32
    V = [[0] * rank for _ in range(m * p)] # 20 x 32
    W = [[0] * rank for _ in range(p * n)] # 10 x 32
    
    def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j, r_off):
        # a_i, a_j: start row, col of A block (2x2)
        # b_i, b_j: start row, col of B block (2x2)
        # c_i, c_j: start row, col of C block (2x2)
        
        # Global indices
        # U: i*m + j
        u00 = a_i * m + a_j
        u01 = a_i * m + a_j + 1
        u10 = (a_i + 1) * m + a_j
        u11 = (a_i + 1) * m + a_j + 1
        
        # V: j*p + k
        v00 = b_i * p + b_j
        v01 = b_i * p + b_j + 1
        v10 = (b_i + 1) * p + b_j
        v11 = (b_i + 1) * p + b_j + 1
        
        # W: k*n + i
        w00 = c_i * n + c_j
        w01 = c_i * n + c_j + 1
        w10 = (c_i + 1) * n + c_j
        w11 = (c_i + 1) * n + c_j + 1
        
        # M1
        U[u00][r_off] = 1
        U[u11][r_off] = 1
        V[v00][r_off] = 1
        V[v11][r_off] = 1
        W[w00][r_off] = 1
        W[w11][r_off] = 1
        
        # M2
        r_off += 1
        U[u10][r_off] = 1
        U[u11][r_off] = 1
        V[v00][r_off] = 1
        W[w10][r_off] = 1
        W[w11][r_off] = -1
        
        # M3
        r_off += 1
        U[u00][r_off] = 1
        V[v10][r_off] = 1
        V[v11][r_off] = -1
        W[w01][r_off] = 1
        W[w11][r_off] = 1
        
        # M4
        r_off += 1
        U[u11][r_off] = 1
        V[v01][r_off] = 1
        V[v00][r_off] = -1
        W[w00][r_off] = 1
        W[w01][r_off] = 1
        
        # M5
        r_off += 1
        U[u00][r_off] = 1
        U[u01][r_off] = 1
        V[v11][r_off] = 1
        W[w00][r_off] = -1
        W[w01][r_off] = 1
        
        # M6
        r_off += 1
        U[u10][r_off] = 1
        U[u00][r_off] = -1
        V[v00][r_off] = 1
        V[v10][r_off] = 1
        W[w11][r_off] = 1
        
        # M7
        r_off += 1
        U[u01][r_off] = 1
        U[u11][r_off] = -1
        V[v11][r_off] = 1
        V[v01][r_off] = 1
        W[w00][r_off] = 1
        
        return r_off + 1

    def add_schoolbook_2x2x1(a_i, a_j, b_i, b_k, c_i, c_k, r_off):
        # A is 2x2 at (a_i, a_j)
        # B is 2x1 at (b_i, b_k)
        # C is 1x2 at (c_i, c_k) -- Wait, C is 5x2.
        # B is 4x5. B block is 2 rows, 1 col.
        # C block is 1 row, 2 cols.
        
        u00 = a_i * m + a_j
        u01 = a_i * m + a_j + 1
        u10 = (a_i + 1) * m + a_j
        u11 = (a_i + 1) * m + a_j + 1
        
        v00 = b_i * p + b_k
        v10 = (b_i + 1) * p + b_k
        
        w00 = c_i * n + c_k
        w01 = c_i * n + c_k + 1
        
        # C0 = A00 B0 + A01 B1
        U[u00][r_off] = 1
        U[u01][r_off] = 1
        V[v00][r_off] = 1
        V[v10][r_off] = 1
        W[w00][r_off] = 1
        
        # C1 = A10 B0 + A11 B1
        r_off += 1
        U[u10][r_off] = 1
        U[u11][r_off] = 1
        V[v00][r_off] = 1
        V[v10][r_off] = 1
        W[w01][r_off] = 1
        
        return r_off + 1

    r = 0
    # A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-1) -> C1 (rows 0-1, cols 0-1)
    r = add_strassen(0, 0, 0, 0, 0, 0, r)
    # A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 2-3) -> C1 (rows 2-3, cols 0-1)
    r = add_strassen(0, 0, 0, 2, 2, 0, r)
    # A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, col 4) -> C1 (row 4, cols 0-1)
    r = add_schoolbook_2x2x1(0, 0, 0, 4, 4, 0, r)
    
    # A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-1) -> C2 (rows 0-1, cols 0-1)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    # A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 2-3) -> C2 (rows 2-3, cols 0-1)
    r = add_strassen(0, 2, 2, 2, 2, 0, r)
    # A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, col 4) -> C2 (row 4, cols 0-1)
    r = add_schoolbook_2x2x1(0, 2, 2, 4, 4, 0, r)
    
    return {"U": U, "V": V, "W": W}

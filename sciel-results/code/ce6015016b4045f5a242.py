"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        x00 = a_rows[0]*m + a_cols[0]
        x01 = a_rows[0]*m + a_cols[1]
        x10 = a_rows[1]*m + a_cols[0]
        x11 = a_rows[1]*m + a_cols[1]
        
        y00 = b_rows[0]*p + b_cols[0]
        y01 = b_rows[0]*p + b_cols[1]
        y10 = b_rows[1]*p + b_cols[0]
        y11 = b_rows[1]*p + b_cols[1]
        
        r00 = c_cols[0]*n + c_rows[0]
        r01 = c_cols[1]*n + c_rows[0]
        r10 = c_cols[0]*n + c_rows[1]
        r11 = c_cols[1]*n + c_rows[1]
        
        # M1 = (x0+x3)(y0+y3) -> r00, r11
        col = offset
        U[x00][col] += 1
        U[x11][col] += 1
        V[y00][col] += 1
        V[y11][col] += 1
        W[r00][col] += 1
        W[r11][col] += 1
        
        # M2 = (x2+x3)(y0) -> r10
        col = offset + 1
        U[x10][col] += 1
        U[x11][col] += 1
        V[y00][col] += 1
        W[r10][col] += 1
        
        # M3 = x0(y1-y3) -> r01
        col = offset + 2
        U[x00][col] += 1
        V[y01][col] += 1
        V[y11][col] += -1
        W[r01][col] += 1
        
        # M4 = x3(y2-y0) -> r00, r10
        col = offset + 3
        U[x11][col] += 1
        V[y10][col] += 1
        V[y00][col] += -1
        W[r00][col] += 1
        W[r10][col] += 1
        
        # M5 = (x0+x1)(y3) -> r01, r11
        col = offset + 4
        U[x00][col] += 1
        U[x01][col] += 1
        V[y11][col] += 1
        W[r01][col] += 1
        W[r11][col] += -1
        
        # M6 = (x2-x0)(y0+y1) -> r11
        col = offset + 5
        U[x10][col] += 1
        U[x00][col] += -1
        V[y00][col] += 1
        V[y01][col] += 1
        W[r11][col] += 1
        
        # M7 = (x1-x3)(y2+y3) -> r00
        col = offset + 6
        U[x01][col] += 1
        U[x11][col] += -1
        V[y10][col] += 1
        V[y11][col] += 1
        W[r00][col] += 1
    
    # Product 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen((0,1), (0,1), (0,1), (0,1), (0,1), (0,1), 0)
    
    # Product 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen((0,1), (0,1), (0,1), (2,3), (0,1), (2,3), 7)
    
    # Product 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen((0,1), (2,3), (2,3), (0,1), (0,1), (0,1), 14)
    
    # Product 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    add_strassen((0,1), (2,3), (2,3), (2,3), (0,1), (2,3), 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

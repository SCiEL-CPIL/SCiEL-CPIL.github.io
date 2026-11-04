import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try Rank 22 first
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_u(i, j, c, val):
        U[i * m + j, c] += val
    def add_v(j, k, c, val):
        V[j * p + k, c] += val
    def add_w(k, i, c, val):
        W[k * n + i, c] += val
        
    def add_strassen_block(a_rows, b_rows, c_cols, offset):
        # m1: (a00+a11)(b00+b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 0, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 0, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 0, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 0, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 0, 1) # C00
        add_w(c_cols[1][0], c_cols[1][1], offset + 0, 1) # C11
        
        # m2: (a10+a11)b00
        add_u(a_rows[2][0], a_rows[2][1], offset + 1, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 1, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 1, 1)
        add_w(c_cols[2][0], c_cols[2][1], offset + 1, 1) # C10
        add_w(c_cols[1][0], c_cols[1][1], offset + 1, -1) # C11
        
        # m3: a00(b01-b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 2, 1)
        add_v(b_rows[1][0], b_rows[1][1], offset + 2, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 2, -1)
        add_w(c_cols[3][0], c_cols[3][1], offset + 2, 1) # C01
        add_w(c_cols[1][0], c_cols[1][1], offset + 2, 1) # C11
        
        # m4: a11(b10-b00)
        add_u(a_rows[3][0], a_rows[3][1], offset + 3, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 3, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 3, -1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 3, 1) # C00
        add_w(c_cols[2][0], c_cols[2][1], offset + 3, 1) # C10
        
        # m5: (a00+a01)b11
        add_u(a_rows[0][0], a_rows[0][1], offset + 4, 1)
        add_u(a_rows[1][0], a_rows[1][1], offset + 4, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 4, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 4, -1) # C00
        add_w(c_cols[3][0], c_cols[3][1], offset + 4, 1) # C01
        
        # m6: (a10-a00)(b00+b10)
        add_u(a_rows[2][0], a_rows[2][1], offset + 5, 1)
        add_u(a_rows[0][0], a_rows[0][1], offset + 5, -1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 5, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 5, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 5, 1) # C11
        
        # m7: (a01-a11)(b10+b11)
        add_u(a_rows[1][0], a_rows[1][1], offset + 6, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 6, -1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 6, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 6, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 6, 1) # C00

    # Block 1: A (rows 0,1; cols 0,1) * B (rows 0,1; cols 0,1)
    a_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    b_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    c_cols_1 = [(0,0), (1,1), (0,1), (1,0)]
    add_strassen_block(a_rows_1, b_rows_1, c_cols_1, 0)
    
    # Block 2: A (rows 0,1; cols 0,1) * B (rows 0,1; cols 2,3)
    b_rows_2 = [(0,2), (0,3), (1,2), (1,3)]
    c_cols_2 = [(2,0), (3,1), (2,1), (3,0)]
    add_strassen_block(a_rows_1, b_rows_2, c_cols_2, 7)
    
    # Block 3: A (rows 0,1; col 2) * B (row 2; cols 0,1,2,3)
    offset = 14
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
            
    # Verify U, V, W
    valid = True
    for k in range(p):
        for i in range(n):
            for j in range(m):
                total = 0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(total, 1.0):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, idx] = 1
                    V_s[j * p + k, idx] = 1
                    W_s[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    
    U_cols = []  # Each element is a list of length n*m
    V_cols = []  # Each element is a list of length m*p
    W_cols = []  # Each element is a list of length p*n
    
    def add_form(u_dict, v_dict, w_dict):
        u_row = [0.0] * (n * m)
        v_row = [0.0] * (m * p)
        w_row = [0.0] * (p * n)
        for (i, j), coeff in u_dict.items():
            u_row[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_row[j * p + k] = coeff
        for (k, i), coeff in w_dict.items():
            w_row[k * n + i] = coeff
        U_cols.append(u_row)
        V_cols.append(v_row)
        W_cols.append(w_row)
    
    def strassen_2x2(a_row, a_col, b_row, b_col):
        # P = A[a_row:a_row+2, a_col:a_col+2]
        # Q = B[b_row:b_row+2, b_col:b_col+2]
        # R = C[a_row:a_row+2, b_col:b_col+2]
        # P[i,j] -> U index (a_row+i, a_col+j)
        # Q[i,j] -> V index (b_row+i, b_col+j)
        # R[i,j] -> W index (b_col+j, a_row+i)
        
        # M1 = (P00+P11)(Q00+Q11)
        add_form(
            {(a_row, a_col): 1, (a_row+1, a_col+1): 1},
            {(b_row, b_col): 1, (b_row+1, b_col+1): 1},
            {(b_col, a_row): 1, (b_col+1, a_row+1): 1}
        )
        
        # M2 = (P10+P11)Q00
        add_form(
            {(a_row+1, a_col): 1, (a_row+1, a_col+1): 1},
            {(b_row, b_col): 1},
            {(b_col, a_row+1): 1, (b_col+1, a_row+1): -1}
        )
        
        # M3 = P00(Q10-Q11)
        add_form(
            {(a_row, a_col): 1},
            {(b_row+1, b_col): 1, (b_row+1, b_col+1): -1},
            {(b_col+1, a_row): 1, (b_col+1, a_row+1): 1}
        )
        
        # M4 = P11(Q01-Q00)
        add_form(
            {(a_row+1, a_col+1): 1},
            {(b_row, b_col+1): 1, (b_row, b_col): -1},
            {(b_col, a_row): 1, (b_col, a_row+1): 1}
        )
        
        # M5 = (P00+P01)Q11
        add_form(
            {(a_row, a_col): 1, (a_row, a_col+1): 1},
            {(b_row+1, b_col+1): 1},
            {(b_col, a_row): -1, (b_col+1, a_row): 1}
        )
        
        # M6 = (P10-P00)(Q00+Q01)
        add_form(
            {(a_row+1, a_col): 1, (a_row, a_col): -1},
            {(b_row, b_col): 1, (b_row, b_col+1): 1},
            {(b_col+1, a_row+1): 1}
        )
        
        # M7 = (P01-P11)(Q10+Q11)
        add_form(
            {(a_row, a_col+1): 1, (a_row+1, a_col+1): -1},
            {(b_row+1, b_col): 1, (b_row+1, b_col+1): 1},
            {(b_col, a_row): 1}
        )
    
    # 4x4x4 part: 8 Strassen multiplications
    strassen_2x2(0, 0, 0, 0)  # A11*B11
    strassen_2x2(0, 2, 2, 0)  # A12*B21
    strassen_2x2(0, 0, 0, 2)  # A11*B12
    strassen_2x2(0, 2, 2, 2)  # A12*B22
    strassen_2x2(2, 0, 0, 0)  # A21*B11
    strassen_2x2(2, 2, 2, 0)  # A22*B21
    strassen_2x2(2, 0, 0, 2)  # A21*B12
    strassen_2x2(2, 2, 2, 2)  # A22*B22
    
    # 4x4x3 part: schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3):
                add_form(
                    {(i, j): 1},
                    {(j, 4 + k): 1},
                    {(4 + k, i): 1}
                )
    
    rank = len(U_cols)
    
    # Build U: (n*m) x rank
    U = [[U_cols[c][i * m + j] for c in range(rank)] for i in range(n) for j in range(m)]
    # Build V: (m*p) x rank
    V = [[V_cols[c][j * p + k] for c in range(rank)] for j in range(m) for k in range(p)]
    # Build W: (p*n) x rank
    W = [[W_cols[c][k * n + i] for c in range(rank)] for k in range(p) for i in range(n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

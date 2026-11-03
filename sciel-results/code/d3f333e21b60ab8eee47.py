import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(i, j, k, val_u, val_v, val_w, col):
        U[i * m + j][col] += val_u
        V[j * p + k][col] += val_v
        W[k * n + i][col] += val_w

    # 1. Strassen for 2x2 core (i in 0,1; j in 0,1; k in 0,1)
    # Note: Python indices are 0-based.
    # A_00, A_01, A_10, A_11
    # B_00, B_01, B_10, B_11
    
    # M1 = A_00 (B_00 + B_11)
    add_term(0, 0, 0, 1, 1, 1, 0) # A00 B00 -> C00
    add_term(0, 0, 1, 1, 1, 1, 0) # A00 B01? No, B_11 is j=1, k=1.
    # Wait, B_11 in 1-based is B[1][1]. In 0-based B[1][1].
    # My add_term uses 0-based i,j,k.
    # B_11 (1-based) is j=1, k=1 (0-based).
    # So add_term(0, 0, 1, 1, 1, 1, 0) adds A_00 B_11 to C_01?
    # C index is k*n + i. k=1, i=0 -> 1*2+0=2. C_01.
    # Correct.
    
    # M2 = (A_00 + A_10) B_11
    add_term(0, 1, 1, 1, 1, 1, 1) # A01? No. A_00 is i=0, j=0. A_10 is i=1, j=0.
    # M2 uses A_00 and A_10.
    add_term(0, 0, 1, 1, 1, 1, 1) # A00 B11 -> C01
    add_term(1, 0, 1, 1, 1, 1, 1) # A10 B11 -> C11
    
    # M3 = (A_01 + A_11) B_00
    add_term(0, 1, 0, 1, 1, 1, 2) # A01 B00 -> C00
    add_term(1, 1, 0, 1, 1, 1, 2) # A11 B00 -> C10
    
    # M4 = A_11 (B_10 + B_00)
    add_term(1, 1, 1, 1, 1, 1, 3) # A11 B10? No B_10 is j=1, k=0.
    add_term(1, 1, 0, 1, 1, 1, 3) # A11 B00 -> C10
    
    # M5 = (A_00 + A_01) B_11
    add_term(0, 0, 1, 1, 1, 1, 4) # A00 B11 -> C01
    add_term(0, 1, 1, 1, 1, 1, 4) # A01 B11 -> C01
    
    # M6 = (A_10 - A_00) (B_00 + B_01)
    add_term(1, 0, 0, 1, 1, 1, 5) # A10 B00 -> C10
    add_term(1, 0, 1, 1, 1, 1, 5) # A10 B01 -> C11
    add_term(0, 0, 0, -1, 1, 1, 5) # -A00 B00 -> -C00
    add_term(0, 0, 1, -1, 1, 1, 5) # -A00 B01 -> -C01
    
    # M7 = (A_01 - A_11) (B_10 + B_11)
    add_term(0, 1, 1, 1, 1, 1, 6) # A01 B11 -> C01
    add_term(0, 1, 0, 1, 1, 1, 6) # A01 B10 -> C00
    add_term(1, 1, 1, -1, 1, 1, 6) # -A11 B11 -> -C11
    add_term(1, 1, 0, -1, 1, 1, 6) # -A11 B10 -> -C10
    
    # 2. A_col3 (j=2) B_row3_k (k=0,1)
    # A_02 B_20 -> C_00
    add_term(0, 2, 0, 1, 1, 1, 7)
    # A_02 B_21 -> C_01
    add_term(0, 2, 1, 1, 1, 1, 8)
    # A_12 B_20 -> C_10
    add_term(1, 2, 0, 1, 1, 1, 9)
    # A_12 B_21 -> C_11
    add_term(1, 2, 1, 1, 1, 1, 10)
    
    # 3. A_core (j=0,1) B_col3 (k=2)
    # A_00 B_02 -> C_02
    add_term(0, 0, 2, 1, 1, 1, 11)
    # A_01 B_12 -> C_02
    add_term(0, 1, 2, 1, 1, 1, 12)
    # A_10 B_02 -> C_12
    add_term(1, 0, 2, 1, 1, 1, 13)
    # A_11 B_12 -> C_12
    add_term(1, 1, 2, 1, 1, 1, 14)
    
    # 4. A_col3 (j=2) B_33 (k=2)
    # A_02 B_22 -> C_02
    add_term(0, 2, 2, 1, 1, 1, 15)
    # A_12 B_22 -> C_12
    add_term(1, 2, 2, 1, 1, 1, 16)
    
    return {"U": U, "V": V, "W": W}

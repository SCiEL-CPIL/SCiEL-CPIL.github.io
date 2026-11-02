import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # U, V are 2x2, W is 2x2
    strassen_terms = [
        ([ [1,0], [0,1] ], [ [1,0], [0,1] ], [ [1,0], [0,1] ]),
        ([ [0,0], [1,1] ], [ [1,0], [0,0] ], [ [0,0], [1,-1] ]),
        ([ [1,0], [0,0] ], [ [0,1], [0,-1] ], [ [0,1], [0,1] ]),
        ([ [0,0], [0,1] ], [ [-1,0], [1,0] ], [ [1,0], [1,0] ]),
        ([ [1,1], [0,0] ], [ [0,0], [0,1] ], [ [-1,1], [0,0] ]),
        ([ [-1,0], [1,0] ], [ [1,1], [0,0] ], [ [0,0], [0,1] ]),
        ([ [0,1], [-1,0] ], [ [0,1], [1,1] ], [ [1,0], [0,0] ])
    ]
    
    # Schoolbook 2x2x1 coefficients
    # U, V are 2x2, 2x1, W is 2x1
    schoolbook_terms = [
        ([ [1,0], [0,0] ], [ [1], [0] ], [ [1], [0] ]),
        ([ [0,1], [0,0] ], [ [0], [1] ], [ [1], [0] ]),
        ([ [0,0], [1,0] ], [ [1], [0] ], [ [0], [1] ]),
        ([ [0,0], [0,1] ], [ [0], [1] ], [ [0], [1] ])
    ]
    
    def add_strassen(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off):
        nonlocal col
        for u_loc, v_loc, w_loc in strassen_terms:
            for i in range(2):
                for j in range(2):
                    if u_loc[i][j] != 0:
                        gi = a_i_off + i
                        gj = a_j_off + j
                        U[gi*m + gj][col] = u_loc[i][j]
            for j in range(2):
                for k in range(2):
                    if v_loc[j][k] != 0:
                        gj = b_j_off + j
                        gk = b_k_off + k
                        V[gj*p + gk][col] = v_loc[j][k]
            for i in range(2):
                for k in range(2):
                    if w_loc[i][k] != 0:
                        gi = c_i_off + i
                        gk = c_k_off + k
                        W[gk*n + gi][col] = w_loc[i][k]
            col += 1

    def add_schoolbook(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off):
        nonlocal col
        for u_loc, v_loc, w_loc in schoolbook_terms:
            for i in range(2):
                for j in range(2):
                    if u_loc[i][j] != 0:
                        gi = a_i_off + i
                        gj = a_j_off + j
                        U[gi*m + gj][col] = u_loc[i][j]
            for j in range(2):
                for k in range(1):
                    if v_loc[j][k] != 0:
                        gj = b_j_off + j
                        gk = b_k_off + k
                        V[gj*p + gk][col] = v_loc[j][k]
            for i in range(2):
                for k in range(1):
                    if w_loc[i][k] != 0:
                        gi = c_i_off + i
                        gk = c_k_off + k
                        W[gk*n + gi][col] = w_loc[i][k]
            col += 1

    # A_L B_T
    # A_L: i in 0,1; j in 0,1
    # B_T: j in 0,1; k in 0,4
    # C: i in 0,1; k in 0,4
    
    # B_T1: k in 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    # B_T2: k in 2,3
    add_strassen(0, 0, 0, 2, 0, 2)
    # B_T3: k in 4
    add_schoolbook(0, 0, 0, 4, 0, 4)
    
    # A_R B_B
    # A_R: i in 0,1; j in 2,3
    # B_B: j in 2,3; k in 0,4
    # C: i in 0,1; k in 0,4
    
    # B_B1: k in 0,1
    add_strassen(0, 2, 2, 0, 0, 0)
    # B_B2: k in 2,3
    add_strassen(0, 2, 2, 2, 0, 2)
    # B_B3: k in 4
    add_schoolbook(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}

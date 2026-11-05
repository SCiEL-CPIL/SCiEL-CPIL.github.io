import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # 6x6 matrices are viewed as 3x3 blocks of 2x2 matrices.
    # Block multiplication is Schoolbook (27 multiplications of 2x2 blocks).
    # Each 2x2 block multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 27 * 7 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrices
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U2: 4x7, V2: 4x7, W2: 4x7
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 1; W2[2][0] = 1; W2[3][0] = 1 # C00+=M1, C01+=M1, C10+=M1, C11+=M1 (Wait, check Strassen formulas)
    
    # Correct Strassen Formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Reset W2 and fill correctly
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] += 1 # C00
    W2[3][0] += 1 # C11
    
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] += 1 # C10
    W2[3][1] += -1 # C11
    
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] += 1 # C01
    W2[3][2] += 1 # C11
    
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] += 1 # C00
    W2[2][3] += 1 # C10
    
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] += -1 # C00
    W2[1][4] += 1 # C01
    
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] += 1 # C11
    
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] += 1 # C00
    
    # Now construct global U, V, W
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # Update U
                    for u_local in range(4):
                        if U2[u_local][s] != 0:
                            # Map local A pos u_local to global row
                            # u_local 0: (0,0) -> row 2*I, col 2*J
                            # u_local 1: (0,1) -> row 2*I, col 2*J+1
                            # u_local 2: (1,0) -> row 2*I+1, col 2*J
                            # u_local 3: (1,1) -> row 2*I+1, col 2*J+1
                            if u_local == 0:
                                row = 2*I
                                col_idx = 2*J
                            elif u_local == 1:
                                row = 2*I
                                col_idx = 2*J+1
                            elif u_local == 2:
                                row = 2*I+1
                                col_idx = 2*J
                            else:
                                row = 2*I+1
                                col_idx = 2*J+1
                            
                            global_row = row * m + col_idx
                            U[global_row][col] += U2[u_local][s]
                            
                    # Update V
                    for v_local in range(4):
                        if V2[v_local][s] != 0:
                            if v_local == 0:
                                row = 2*J
                                col_idx = 2*K
                            elif v_local == 1:
                                row = 2*J
                                col_idx = 2*K+1
                            elif v_local == 2:
                                row = 2*J+1
                                col_idx = 2*K
                            else:
                                row = 2*J+1
                                col_idx = 2*K+1
                            
                            global_row = row * p + col_idx
                            V[global_row][col] += V2[v_local][s]
                            
                    # Update W
                    for w_local in range(4):
                        if W2[w_local][s] != 0:
                            if w_local == 0:
                                row = 2*I
                                col_idx = 2*K
                            elif w_local == 1:
                                row = 2*I
                                col_idx = 2*K+1
                            elif w_local == 2:
                                row = 2*I+1
                                col_idx = 2*K
                            else:
                                row = 2*I+1
                                col_idx = 2*K+1
                            
                            # W index convention: W[k*n+i, r]
                            # Here n=6 (output cols), p=6 (input cols of B? No, W is p*n)
                            # W shape is (p*n, rank).
                            # Index is k*n + i.
                            # In our block C_IK, i is row, k is col.
                            # So global row index for W is k * n + i.
                            # k corresponds to K block index (0..2), i to I block index (0..2).
                            # w_local maps to (i_local, k_local).
                            # w_local 0: i=0, k=0
                            # w_local 1: i=0, k=1
                            # w_local 2: i=1, k=0
                            # w_local 3: i=1, k=1
                            
                            if w_local == 0:
                                i_global = 2*I
                                k_global = 2*K
                            elif w_local == 1:
                                i_global = 2*I
                                k_global = 2*K+1
                            elif w_local == 2:
                                i_global = 2*I+1
                                k_global = 2*K
                            else:
                                i_global = 2*I+1
                                k_global = 2*K+1
                                
                            global_w_row = k_global * n + i_global
                            W[global_w_row][col] += W2[w_local][s]
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

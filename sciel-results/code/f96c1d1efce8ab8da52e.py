import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Construct Strassen 2x2x2 decomposition (Rank 7)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # W index: k*2 + i
    
    U_inner = [[0]*7 for _ in range(4)]
    V_inner = [[0]*7 for _ in range(4)]
    W_inner = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U_inner[0][0] = 1; U_inner[3][0] = 1
    V_inner[0][0] = 1; V_inner[3][0] = 1
    W_inner[0][0] = 1; W_inner[3][0] = 1 # C00 (0*2+0), C11 (1*2+1)
    
    # M2: (A10+A11)B00 -> C10
    U_inner[2][1] = 1; U_inner[3][1] = 1
    V_inner[0][1] = 1
    W_inner[2][1] = 1 # C10 (0*2+1? No, C row 1 col 0. k=0, i=1 -> 0*2+1=1? No.
    # W index is k*n + i. n=2.
    # C[i][k].
    # C10 means i=1, k=0. Index = 0*2 + 1 = 1.
    # Wait, standard matrix C[i][j]. Here C[i][k].
    # Task says W[k*n+i].
    # So for C10 (row 1, col 0), k=0, i=1. Index = 0*2+1 = 1.
    # My previous mapping: W_inner[2][1] = 1. Index 2 is k=1, i=0. That is C01.
    # Let's re-verify Strassen formulas with indices.
    
    # Let's restart the Strassen mapping carefully.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # W index: k*2 + i.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    
    U_inner = [[0]*7 for _ in range(4)]
    V_inner = [[0]*7 for _ in range(4)]
    W_inner = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # A00 (0), A11 (3)
    # B00 (0), B11 (3)
    # Contributes to C00 (0) and C11 (3)
    U_inner[0][0] = 1; U_inner[3][0] = 1
    V_inner[0][0] = 1; V_inner[3][0] = 1
    W_inner[0][0] = 1; W_inner[3][0] = 1
    
    # M2: (A10+A11)B00
    # A10 (2), A11 (3)
    # B00 (0)
    # Contributes to C10 (1)
    U_inner[2][1] = 1; U_inner[3][1] = 1
    V_inner[0][1] = 1
    W_inner[1][1] = 1
    
    # M3: A00(B10-B11)
    # A00 (0)
    # B10 (2), B11 (3)
    # Contributes to C01 (2)
    U_inner[0][2] = 1
    V_inner[2][2] = 1; V_inner[3][2] = -1
    W_inner[2][2] = 1
    
    # M4: A11(B01-B00)
    # A11 (3)
    # B01 (1), B00 (0)
    # Contributes to C00 (0)
    U_inner[3][3] = 1
    V_inner[1][3] = 1; V_inner[0][3] = -1
    W_inner[0][3] = 1
    
    # M5: (A00+A01)B11
    # A00 (0), A01 (1)
    # B11 (3)
    # Contributes to C00 (0) with -1
    U_inner[0][4] = 1; U_inner[1][4] = 1
    V_inner[3][4] = 1
    W_inner[0][4] = -1
    
    # M6: (A10-A00)(B00+B01)
    # A10 (2), A00 (0)
    # B00 (0), B01 (1)
    # Contributes to C11 (3)
    U_inner[2][5] = 1; U_inner[0][5] = -1
    V_inner[0][5] = 1; V_inner[1][5] = 1
    W_inner[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # A01 (1), A11 (3)
    # B10 (2), B11 (3)
    # Contributes to C00 (0) with +1
    U_inner[1][6] = 1; U_inner[3][6] = -1
    V_inner[2][6] = 1; V_inner[3][6] = 1
    W_inner[0][6] = 1
    
    if time.time() > deadline:
        return {"U": [[0]*216 for _ in range(36)], "V": [[0]*216 for _ in range(36)], "W": [[0]*216 for _ in range(36)]}

    # 2. Construct Standard 3x3x3 decomposition (Rank 27)
    # Indices: I in {0,1,2}, J in {0,1,2}, K in {0,1,2}
    # U_outer shape (9, 27), V_outer shape (9, 27), W_outer shape (9, 27)
    # W_outer index: K*3 + I
    
    U_outer = [[0]*27 for _ in range(9)]
    V_outer = [[0]*27 for _ in range(9)]
    W_outer = [[0]*27 for _ in range(9)]
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # A[I][J] * B[J][K] -> C[I][K]
                U_outer[I*3 + J][col] = 1
                V_outer[J*3 + K][col] = 1
                W_outer[K*3 + I][col] = 1
                col += 1
    
    if time.time() > deadline:
        return {"U": [[0]*216 for _ in range(36)], "V": [[0]*216 for _ in range(36)], "W": [[0]*216 for _ in range(36)]}

    # 3. Combine
    rank_outer = 27
    rank_inner = 7
    total_rank = rank_outer * rank_inner
    
    U = [[0]*total_rank for _ in range(36)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    for r_out in range(rank_outer):
        for r_in in range(rank_inner):
            r_total = r_out * rank_inner + r_in
            
            for I in range(3):
                u_val = U_outer[I][r_out]
                if u_val != 0:
                    for u in range(2):
                        u_in_val = U_inner[u][r_in]
                        if u_in_val != 0:
                            i = 2*I + u
                            U[i][r_total] = u_val * u_in_val
            
            for J in range(3):
                v_val = V_outer[J][r_out]
                if v_val != 0:
                    for v in range(2):
                        v_in_val = V_inner[v][r_in]
                        if v_in_val != 0:
                            j = 2*J + v
                            V[j][r_total] = v_val * v_in_val
            
            for K in range(3):
                for I in range(3):
                    w_out_val = W_outer[K*3 + I][r_out]
                    if w_out_val != 0:
                        for w in range(2):
                            for u in range(2):
                                w_in_val = W_inner[w*2 + u][r_in]
                                if w_in_val != 0:
                                    # W index: k*6 + i
                                    # k = 2K + w
                                    # i = 2I + u
                                    k = 2*K + w
                                    i = 2*I + u
                                    w_idx = k*6 + i
                                    W[w_idx][r_total] = w_out_val * w_in_val
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

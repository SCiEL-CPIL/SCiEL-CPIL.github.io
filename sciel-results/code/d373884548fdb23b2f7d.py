import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 36: 4 * 7 (Strassen) + 8 (Schoolbook for k=4)
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen tensors for 2x2x2
    # U: (i, j), V: (j, k), W: (k, i)
    # Flattened: U: i*2+j, V: j*2+k, W: k*2+i
    strassen_U = [
        [1, 0, 0, 1], # P1: A00+A11
        [0, 0, 1, 1], # P2: A10+A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00+A10
        [0, 1, 0, -1],# P6: A01-A11
        [-1, 1, 0, 0] # P7: A10-A00 (Wait, P7 is A10-A00? Yes)
    ]
    # Check P7: (A10 - A00). U(1,0)=1, U(0,0)=-1.
    # My list: U[6] = [-1, 1, 0, 0]. Index 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1).
    # So U[6][0] = -1 (A00), U[6][2] = 1 (A10). Correct.
    
    strassen_V = [
        [1, 0, 0, 1], # P1: B00+B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1],# P3: B01-B11
        [0, -1, 1, 0],# P4: B10-B00
        [0, 0, 0, 1], # P5: B11
        [0, 0, 1, 1], # P6: B10+B11
        [1, 1, 0, 0]  # P7: B00+B01
    ]
    # Check P4: B10-B00. V(1,0)=1, V(0,0)=-1.
    # V index: j*2+k. (1,0)->2, (0,0)->0.
    # My list: V[3] = [0, -1, 1, 0]. Index 0 is -1, Index 2 is 1. Correct.
    
    strassen_W = [
        [1, 0, 0, 1], # P1: C00, C11
        [0, 1, 0, 1], # P2: C10, C11
        [0, 0, 1, 0], # P3: C01? P3 is A00(B01-B11). Contributes to C01.
                      # C01 -> k=1, i=0 -> W(1,0) -> index 2.
                      # My list: W[2] = [0, 0, 1, 0]. Index 2 is 1. Correct.
        [1, 0, 0, 1], # P4: C10, C11?
                      # P4 is A11(B10-B00). Contributes to C10, C11.
                      # C10 -> k=0, i=1 -> W(0,1) -> index 1.
                      # C11 -> k=1, i=1 -> W(1,1) -> index 3.
                      # My list: W[3] = [1, 0, 0, 1]. Index 1 is 0?
                      # Wait. W index: k*2+i.
                      # (0,1) -> 1. (1,1) -> 3.
                      # So W[3] should be [0, 1, 0, 1].
                      # Let's re-evaluate W.
    ]
    
    # Let's carefully define W based on C contributions.
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    # W for M1 (P1): C00, C11. W(0,0)=1, W(1,1)=1. -> [1, 0, 0, 1]
    # W for M2 (P2): C10, C11. W(0,1)=1, W(1,1)=1. -> [0, 1, 0, 1]
    # W for M3 (P3): C01, C11. W(1,0)=1, W(1,1)=1. -> [0, 0, 1, 1]
    # W for M4 (P4): C00, C10. W(0,0)=1, W(0,1)=1. -> [1, 1, 0, 0]
    # W for M5 (P5): C00, C01. W(0,0)=1, W(1,0)=1. -> [1, 0, 1, 0]
    # W for M6 (P6): C11. W(1,1)=1. -> [0, 0, 0, 1]
    # W for M7 (P7): C00. W(0,0)=1. -> [1, 0, 0, 0]
    
    strassen_W = [
        [1, 0, 0, 1], # P1
        [0, 1, 0, 1], # P2
        [0, 0, 1, 1], # P3
        [1, 1, 0, 0], # P4
        [1, 0, 1, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]

    col = 0
    # 4 Strassen blocks
    # j_offsets: 0, 2
    # k_offsets: 0, 2
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for s in range(7):
                # Map U
                for i in range(2):
                    for j_loc in range(2):
                        val = strassen_U[s][i*2 + j_loc]
                        if val != 0:
                            i_glob = i
                            j_glob = j_off + j_loc
                            U[i_glob * m + j_glob][col] += val
                
                # Map V
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = strassen_V[s][j_loc*2 + k_loc]
                        if val != 0:
                            j_glob = j_off + j_loc
                            k_glob = k_off + k_loc
                            V[j_glob * p + k_glob][col] += val
                
                # Map W
                for k_loc in range(2):
                    for i in range(2):
                        val = strassen_W[s][k_loc*2 + i]
                        if val != 0:
                            k_glob = k_off + k_loc
                            i_glob = i
                            W[k_glob * n + i_glob][col] += val
                
                col += 1

    # Schoolbook for k=4
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 8
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

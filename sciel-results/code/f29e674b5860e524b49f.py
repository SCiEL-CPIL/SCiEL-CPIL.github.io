import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strategy: Karatsuba-style on intermediate dimension m=4
    # A (2x4) = [A0, A1] where A0, A1 are 2x2
    # B (4x8) = [B0; B1] where B0, B1 are 2x8
    # C = A0*B0 + A1*B1
    # Each Ai*Bi is 2x2 * 2x8, decomposed as 4 independent 2x2*2x2 Strassen (rank 7 each) = rank 28
    # Total rank = 2 * 28 = 56
    
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]   # 16 x 56
    
    # Strassen 2x2 local factorization (rank 7)
    # Local indices: A: a00=0, a01=1, a10=2, a11=3
    #                B: b00=0, b01=1, b10=2, b11=3
    #                C: c00=0, c01=1, c10=2, c11=3
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]
    
    # r=0: M1=(a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0] += 1; U_loc[3][0] += 1
    V_loc[0][0] += 1; V_loc[3][0] += 1
    W_loc[0][0] += 1; W_loc[3][0] += 1
    # r=1: M2=(a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1] += 1; U_loc[3][1] += 1
    V_loc[0][1] += 1
    W_loc[2][1] += 1; W_loc[3][1] += -1
    # r=2: M3=a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2] += 1
    V_loc[1][2] += 1; V_loc[3][2] += -1
    W_loc[1][2] += 1; W_loc[3][2] += 1
    # r=3: M4=a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3] += 1
    V_loc[2][3] += 1; V_loc[0][3] += -1
    W_loc[0][3] += 1; W_loc[2][3] += 1
    # r=4: M5=(a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4] += 1; U_loc[1][4] += 1
    V_loc[3][4] += 1
    W_loc[0][4] += -1; W_loc[1][4] += 1
    # r=5: M6=(a10-a00)(b00+b01) -> c11+1
    U_loc[2][5] += 1; U_loc[0][5] += -1
    V_loc[0][5] += 1; V_loc[1][5] += 1
    W_loc[3][5] += 1
    # r=6: M7=(a01-a11)(b10+b11) -> c00+1
    U_loc[1][6] += 1; U_loc[3][6] += -1
    V_loc[2][6] += 1; V_loc[3][6] += 1
    W_loc[0][6] += 1
    
    # For each half h (0 or 1) of the m-dimension, and each column-pair jj (0..3):
    # A-half h: rows 0,1 of A, columns 2h, 2h+1 -> global A indices: i*m + 2h + di
    # B-half h: rows 2h, 2h+1 of B, columns 2jj, 2jj+1 -> global B indices: (2h+dr)*p + 2jj + dk
    # C: columns 2jj, 2jj+1 -> global W index: (2jj+dk)*n + i
    
    group = 0
    for h in range(2):  # two halves of m
        for jj in range(4):  # four column-pairs of p
            base_r = group * 7
            group += 1
            
            # A local to global mapping
            # local a00 -> A[0, 2h] = 0*4 + 2h
            # local a01 -> A[0, 2h+1] = 0*4 + 2h+1
            # local a10 -> A[1, 2h] = 1*4 + 2h
            # local a11 -> A[1, 2h+1] = 1*4 + 2h+1
            a_map = [
                0 * m + (2 * h + 0),
                0 * m + (2 * h + 1),
                1 * m + (2 * h + 0),
                1 * m + (2 * h + 1),
            ]
            
            # B local to global mapping
            # local b00 -> B[2h, 2jj] = (2h)*p + 2jj
            # local b01 -> B[2h, 2jj+1] = (2h)*p + 2jj+1
            # local b10 -> B[2h+1, 2jj] = (2h+1)*p + 2jj
            # local b11 -> B[2h+1, 2jj+1] = (2h+1)*p + 2jj+1
            b_map = [
                (2 * h + 0) * p + (2 * jj + 0),
                (2 * h + 0) * p + (2 * jj + 1),
                (2 * h + 1) * p + (2 * jj + 0),
                (2 * h + 1) * p + (2 * jj + 1),
            ]
            
            # C local to global W mapping
            # local c00 -> C[0, 2jj] -> W[2jj*n + 0]
            # local c01 -> C[0, 2jj+1] -> W[(2jj+1)*n + 0]
            # local c10 -> C[1, 2jj] -> W[2jj*n + 1]
            # local c11 -> C[1, 2jj+1] -> W[(2jj+1)*n + 1]
            c_map = [
                (2 * jj + 0) * n + 0,
                (2 * jj + 1) * n + 0,
                (2 * jj + 0) * n + 1,
                (2 * jj + 1) * n + 1,
            ]
            
            for s in range(7):
                r = base_r + s
                for la in range(4):
                    val = U_loc[la][s]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_loc[lb][s]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_loc[lc][s]
                    if val != 0:
                        W[c_map[lc]][r] += val
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    
    # Strategy:
    # 1. Use Strassen's algorithm for the 2x2x2 sub-block (i,j,k in {0,1}).
    #    This reduces rank from 8 to 7 for that block.
    # 2. Use schoolbook for the remaining 67 triples.
    # Total rank: 7 + 67 = 74.
    
    rank_strassen = 7
    remaining_triples = n * m * p - 2 * 2 * 2
    total_rank = rank_strassen + remaining_triples
    
    U = np.zeros((n * m, total_rank), dtype=object)
    V = np.zeros((m * p, total_rank), dtype=object)
    W = np.zeros((p * n, total_rank), dtype=object)
    
    # --- Part 1: Strassen for i,j,k in {0,1} ---
    # Indices in U: i*m + j. For i,j in {0,1}: 0,1,5,6 (since m=5)
    # Indices in V: j*p + k. For j,k in {0,1}: 0,1,5,6 (since p=5)
    # Indices in W: k*n + i. For k,i in {0,1}: 0,1,3,4 (since n=3)
    
    # Strassen coefficients
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U coefficients (u_ij)
    # P1: A00, A11
    U[0*5+0, 0] = 1
    U[1*5+1, 0] = 1
    
    # P2: A10, A11
    U[1*5+0, 1] = 1
    U[1*5+1, 1] = 1
    
    # P3: A00
    U[0*5+0, 2] = 1
    
    # P4: A11
    U[1*5+1, 3] = 1
    
    # P5: A00, A01
    U[0*5+0, 4] = 1
    U[0*5+1, 4] = 1
    
    # P6: A10, A00
    U[1*5+0, 5] = 1
    U[0*5+0, 5] = -1
    
    # P7: A01, A11
    U[0*5+1, 6] = 1
    U[1*5+1, 6] = -1
    
    # V coefficients (v_jk)
    # P1: B00, B11
    V[0*5+0, 0] = 1
    V[1*5+1, 0] = 1
    
    # P2: B00
    V[0*5+0, 1] = 1
    
    # P3: B01, B11
    V[0*5+1, 2] = 1
    V[1*5+1, 2] = -1
    
    # P4: B10, B00
    V[1*5+0, 3] = 1
    V[0*5+0, 3] = -1
    
    # P5: B11
    V[1*5+1, 4] = 1
    
    # P6: B00, B01
    V[0*5+0, 5] = 1
    V[0*5+1, 5] = 1
    
    # P7: B10, B11
    V[1*5+0, 6] = 1
    V[1*5+1, 6] = 1
    
    # W coefficients (w_ki)
    # C00 = P1 + P4 - P5 + P7
    W[0*3+0, 0] = 1
    W[0*3+0, 3] = 1
    W[0*3+0, 4] = -1
    W[0*3+0, 6] = 1
    
    # C01 = P3 + P5
    W[0*3+1, 2] = 1
    W[0*3+1, 4] = 1
    
    # C10 = P2 + P4
    W[1*3+0, 1] = 1
    W[1*3+0, 3] = 1
    
    # C11 = P1 - P2 + P3 + P6
    W[1*3+1, 0] = 1
    W[1*3+1, 1] = -1
    W[1*3+1, 2] = 1
    W[1*3+1, 5] = 1
    
    # --- Part 2: Schoolbook for remaining triples ---
    # Triples (i, j, k) where NOT (i<2 and j<2 and k<2)
    col = rank_strassen
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

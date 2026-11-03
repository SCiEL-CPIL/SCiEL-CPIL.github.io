import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We use a Strassen-based decomposition with rank 56.
    # The key insight: split the middle dimension m=4 into two groups of 2,
    # and for each group, use 4 independent 2x2*2x2 Strassen decompositions
    # (one for each pair of output columns in the 2x8 result).
    
    # Strassen components for 2x2 matrix multiplication
    # U_local: (4, 7) - maps 4 input elements of A to 7 Strassen products
    # V_local: (4, 7) - maps 4 input elements of B to 7 Strassen products  
    # W_local: (4, 7) - maps 4 output elements of C from 7 Strassen products
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1): M1 = (A00+A11)(B00+B11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # r=1 (M2): M2 = (A10+A11)B00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # r=2 (M3): M3 = A00(B10-B00)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # r=3 (M4): M4 = A11(B11-B01)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # r=4 (M5): M5 = (A00+A01)B11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # r=5 (M6): M6 = (A10-A00)(B00+B01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7): M7 = (A01-A11)(B10+B11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # For each of the 4 column pairs (j=0,1,2,3) and 2 row groups (s=0,1):
    # We decompose a 2x2 * 2x2 multiplication using Strassen
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            if s == 0:
                # A block: A[i, j] for i in {0,1}, j in {0,1}
                # Global A index: i * 4 + j
                a_map = [0*4 + 0, 0*4 + 1, 1*4 + 0, 1*4 + 1]
            else:
                # A block: A[i, j] for i in {0,1}, j in {2,3}
                # Global A index: i * 4 + j
                a_map = [0*4 + 2, 0*4 + 3, 1*4 + 2, 1*4 + 3]
            
            # B block: B[j, k] for j in {2s, 2s+1}, k in {2j, 2j+1}
            # Global B index: j * 8 + k
            if s == 0:
                b_map = [0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1]
            else:
                b_map = [2*8 + 2*j, 2*8 + 2*j+1, 3*8 + 2*j, 3*8 + 2*j+1]
            
            # C block: C[i, k] for i in {0,1}, k in {2j, 2j+1}
            # Global C index: k * n + i = k * 2 + i
            c_map = [4*j + 0, 4*j + 2, 4*j + 1, 4*j + 3]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc], r] += val
                        
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Safety margin: return what we have
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

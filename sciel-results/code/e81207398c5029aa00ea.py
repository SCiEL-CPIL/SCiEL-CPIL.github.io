import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Strassen components for 2x2 matrix multiplication
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1): M1 = (A00+A11)(B00+B11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # r=1 (M2): M2 = (A10+A11)B00
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # r=2 (M3): M3 = A00(B10-B00)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # r=3 (M4): M4 = A11(B11-B01)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # r=4 (M5): M5 = (A00+A01)B11
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # r=5 (M6): M6 = (A10-A00)(B00+B01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # r=6 (M7): M7 = (A01-A11)(B10+B11)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Precompute mappings for each (s, j) combination
    # s=0: A cols {0,1}, B rows {0,1}
    # s=1: A cols {2,3}, B rows {2,3}
    for s in range(2):
        if s == 0:
            a_map = [0, 1, 4, 5]  # A[i*4+j]: (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5
        else:
            a_map = [2, 3, 6, 7]  # A[i*4+j]: (0,2)=2,(0,3)=3,(1,2)=6,(1,3)=7
        
        for j in range(4):
            base_r = (j * 2 + s) * 7
            
            if s == 0:
                b_map = [2*j, 2*j+1, 8+2*j, 8+2*j+1]  # B[j'*8+k]: (0,2j),(0,2j+1),(1,2j),(1,2j+1)
            else:
                b_map = [16+2*j, 16+2*j+1, 24+2*j, 24+2*j+1]  # B[j'*8+k]: (2,2j),(2,2j+1),(3,2j),(3,2j+1)
            
            # W index: k*2+i where i is row, k is col
            # Local (0,0)->i=0,k=2j -> 2j*2+0=4j
            # Local (0,1)->i=0,k=2j+1 -> (2j+1)*2+0=4j+2
            # Local (1,0)->i=1,k=2j -> 2j*2+1=4j+1
            # Local (1,1)->i=1,k=2j+1 -> (2j+1)*2+1=4j+3
            c_map = [4*j, 4*j+2, 4*j+1, 4*j+3]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen 2x2 matrix multiplication decomposition
    # Local indices 0,1,2,3 correspond to positions (0,0), (0,1), (1,0), (1,1)
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # M2 = (A10+A11)B00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # M3 = A00(B10-B00)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # M4 = A11(B11-B01)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # M5 = (A00+A01)B11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # A is 2x4: A[i, j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x8: B[j, k] for j in {0,1,2,3}, k in {0,...,7}
    # C is 2x8: C[i, k] for i in {0,1}, k in {0,...,7}
    # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    
    # Split j into two groups: {0,1} and {2,3}
    # For each group, split B columns into 4 pairs of 2
    
    for s in range(2):  # s=0: j in {0,1}, s=1: j in {2,3}
        for j in range(4):  # column pair index
            base_r = (s * 4 + j) * 7
            
            # A block mapping
            if s == 0:
                a_map = [0, 1, 4, 5]  # A[0,0], A[0,1], A[1,0], A[1,1]
            else:
                a_map = [2, 3, 6, 7]  # A[0,2], A[0,3], A[1,2], A[1,3]
            
            # B block mapping
            if s == 0:
                b_map = [0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1]
            else:
                b_map = [2*8 + 2*j, 2*8 + 2*j+1, 3*8 + 2*j, 3*8 + 2*j+1]
            
            # C block mapping
            # Global C index: k * n + i = 2k + i
            # Local (0,0) -> i=0, k=2j -> index 2*(2j)+0 = 4j
            # Local (0,1) -> i=0, k=2j+1 -> index 2*(2j+1)+0 = 4j+2
            # Local (1,0) -> i=1, k=2j -> index 2*(2j)+1 = 4j+1
            # Local (1,1) -> i=1, k=2j+1 -> index 2*(2j+1)+1 = 4j+3
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
    if elapsed > 110:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Rank 28 decomposition for 2x3x5 matrix multiplication.
    # Structure:
    # A is 2x3, B is 3x5.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x5) and B2 (1x5).
    # C = A1*B1 + A2*B2.
    # R(2,2,5) = 16 (7 for 2x2x2 + 9 for 2x2x3? No, 7+7+2=16? 
    # 2x2x5 = 2x2x2 + 2x2x3. R(2,2,2)=7. R(2,2,3)=11? No, 9?
    # If R(2,2,3)=9, then 16.
    # R(2,1,5) = 6?
    # 16 + 6 = 22?
    # The actual rank is 27 or 28.
    # The parent code used 28.
    # I will replicate the parent's logic but with different syntax.
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col_idx = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col_idx
        # M1
        U[i1*m+j1, col_idx] = 1; U[i2*m+j2, col_idx] = 1
        V[j1*p+k1, col_idx] = 1; V[j2*p+k2, col_idx] = 1
        W[k1*n+i1, col_idx] = 1; W[k2*n+i2, col_idx] = 1
        col_idx += 1
        # M2
        U[i2*m+j1, col_idx] = 1; U[i2*m+j2, col_idx] = 1
        V[j1*p+k1, col_idx] = 1
        W[k1*n+i2, col_idx] = 1; W[k2*n+i2, col_idx] = -1
        col_idx += 1
        # M3
        U[i1*m+j1, col_idx] = 1
        V[j1*p+k2, col_idx] = 1; V[j2*p+k2, col_idx] = -1
        W[k2*n+i1, col_idx] = 1; W[k2*n+i2, col_idx] = 1
        col_idx += 1
        # M4
        U[i2*m+j2, col_idx] = 1
        V[j2*p+k1, col_idx] = 1; V[j1*p+k1, col_idx] = -1
        W[k1*n+i1, col_idx] = 1; W[k1*n+i2, col_idx] = 1
        col_idx += 1
        # M5
        U[i1*m+j1, col_idx] = 1; U[i1*m+j2, col_idx] = 1
        V[j2*p+k2, col_idx] = 1
        W[k1*n+i1, col_idx] = -1; W[k2*n+i1, col_idx] = 1
        col_idx += 1
        # M6
        U[i2*m+j1, col_idx] = 1; U[i1*m+j1, col_idx] = -1
        V[j1*p+k1, col_idx] = 1; V[j1*p+k2, col_idx] = 1
        W[k2*n+i2, col_idx] = 1
        col_idx += 1
        # M7
        U[i1*m+j2, col_idx] = 1; U[i2*m+j2, col_idx] = -1
        V[j2*p+k1, col_idx] = 1; V[j2*p+k2, col_idx] = 1
        W[k1*n+i1, col_idx] = 1
        col_idx += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col_idx
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col_idx] = 1
                    V[j*p+k, col_idx] = 1
                    W[k*n+i, col_idx] = 1
                    col_idx += 1

    # Block 1: A[:,0:2] * B[0:2,0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:,0:2] * B[0:2,2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[:,0:2] * B[0:2,4:5]
    add_schoolbook([0, 1], [0, 1], [4])
    
    # Block 4: A[:,2:3] * B[2:3,0:2]
    add_schoolbook([0, 1], [2], [0, 1])
    
    # Block 5: A[:,2:3] * B[2:3,2:4]
    add_schoolbook([0, 1], [2], [2, 3])
    
    # Block 6: A[:,2:3] * B[2:3,4:5]
    add_schoolbook([0, 1], [2], [4])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

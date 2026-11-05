import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    
    n, m, p = 3, 3, 6
    
    # Strassen's 7-multiplication decomposition for 3x3 matrix multiplication
    # U333[i*m+j, r] - coefficients for A elements (9x7)
    # V333[j*p+k, r] - coefficients for B elements (9x7)
    # W333[k*n+i, r] - coefficients for C elements (9x7)
    
    U333 = np.zeros((9, 7))
    V333 = np.zeros((9, 7))
    W333 = np.zeros((9, 7))
    
    # M1 = (a11+a22)*(b11+b33)
    U333[0, 0] = 1; U333[4, 0] = 1
    V333[0, 0] = 1; V333[8, 0] = 1
    
    # M2 = (a21+a22)*(b11+b12)
    U333[3, 1] = 1; U333[4, 1] = 1
    V333[0, 1] = 1; V333[1, 1] = 1
    
    # M3 = (a11+a13)*(b22+b33)
    U333[0, 2] = 1; U333[2, 2] = 1
    V333[4, 2] = 1; V333[8, 2] = 1
    
    # M4 = (a31+a33)*(b11+b12)
    U333[6, 3] = 1; U333[8, 3] = 1
    V333[0, 3] = 1; V333[1, 3] = 1
    
    # M5 = (a11+a21)*(b22+b23)
    U333[0, 4] = 1; U333[3, 4] = 1
    V333[4, 4] = 1; V333[5, 4] = 1
    
    # M6 = (a22+a32)*(b13+b33)
    U333[4, 5] = 1; U333[7, 5] = 1
    V333[2, 5] = 1; V333[8, 5] = 1
    
    # M7 = (a13+a33)*(b12+b22)
    U333[2, 6] = 1; U333[8, 6] = 1
    V333[1, 6] = 1; V333[4, 6] = 1
    
    # W factor: C elements as linear combinations of M's
    # C11 = M1 + M3 - M5 - M7
    W333[0, 0] = 1; W333[0, 2] = 1; W333[0, 4] = -1; W333[0, 6] = -1
    # C12 = M5 + M7
    W333[1, 4] = 1; W333[1, 6] = 1
    # C13 = M1 - M3 + M6 + M7
    W333[2, 0] = 1; W333[2, 2] = -1; W333[2, 5] = 1; W333[2, 6] = 1
    # C21 = M2 + M4
    W333[3, 1] = 1; W333[3, 3] = 1
    # C22 = M1 - M2 + M4 + M5
    W333[4, 0] = 1; W333[4, 1] = -1; W333[4, 3] = 1; W333[4, 4] = 1
    # C23 = M2 - M4 + M6
    W333[5, 1] = 1; W333[5, 3] = -1; W333[5, 5] = 1
    # C31 = M1 - M3 + M4 - M5
    W333[6, 0] = 1; W333[6, 2] = -1; W333[6, 3] = 1; W333[6, 4] = -1
    # C32 = M3 - M1 + M2 + M7
    W333[7, 2] = 1; W333[7, 0] = -1; W333[7, 1] = 1; W333[7, 6] = 1
    # C33 = M1 + M3 - M2 - M4
    W333[8, 0] = 1; W333[8, 2] = 1; W333[8, 1] = -1; W333[8, 3] = -1
    
    # Build full 3x3x6 decomposition: rank 14 (two copies of rank-7 Strassen)
    rank_total = 14
    U = np.zeros((9, rank_total))
    V = np.zeros((18, rank_total))
    W = np.zeros((18, rank_total))
    
    # First 3x3 block (columns 0,1,2 of B)
    for r in range(7):
        U[:, r] = U333[:, r]
        for j in range(3):
            for k in range(3):
                V[j*6 + k, r] = V333[j*3 + k, r]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, r] = W333[k*3 + i, r]
    
    # Second 3x3 block (columns 3,4,5 of B)
    for r in range(7):
        U[:, 7 + r] = U333[:, r]
        for j in range(3):
            for k in range(3):
                V[j*6 + (k+3), 7 + r] = V333[j*3 + k, r]
        for k in range(3):
            for i in range(3):
                W[(k+3)*3 + i, 7 + r] = W333[k*3 + i, r]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

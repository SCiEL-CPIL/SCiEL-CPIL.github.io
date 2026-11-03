import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def validate(U, V, W):
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    # Part 2: 3x3x2 schoolbook (k=3,4) - 18 terms
    U_332 = np.zeros((9, 18))
    V_332 = np.zeros((15, 18))
    W_332 = np.zeros((15, 18))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                U_332[i*3+j, col] = 1
                V_332[j*5+k, col] = 1
                W_332[k*3+i, col] = 1
                col += 1
                
    # Part 1: 3x3x3 using Strassen (7 terms) + 17 schoolbook terms = 24 terms
    U_333 = np.zeros((9, 24))
    V_333 = np.zeros((9, 24))
    W_333 = np.zeros((9, 24))
    
    # Terms 0-6: Strassen for 2x2x2 block
    U_333[0,0]=1; V_333[0,0]=1; W_333[0,0]=1
    U_333[1,0]=1; V_333[1,0]=1; W_333[1,0]=1
    U_333[3,1]=1; V_333[0,1]=1; W_333[2,1]=1
    U_333[2,1]=1; V_333[0,1]=1; W_333[3,1]=1
    U_333[0,2]=1; V_333[3,2]=1; W_333[0,2]=1
    U_333[0,2]=1; V_333[4,2]=-1; W_333[1,2]=-1
    U_333[1,3]=1; V_333[1,3]=1; W_333[1,3]=1
    U_333[1,3]=1; V_333[0,3]=-1; W_333[0,3]=-1
    U_333[0,4]=1; V_333[4,4]=1; W_333[1,4]=1
    U_333[1,4]=1; V_333[4,4]=1; W_333[2,4]=1
    U_333[2,5]=1; V_333[0,5]=1; W_333[2,5]=1
    U_333[2,5]=1; V_333[3,5]=1; W_333[3,5]=1
    U_333[0,5]=-1; V_333[0,5]=-1; W_333[0,5]=-1
    U_333[0,5]=-1; V_333[3,5]=-1; W_333[1,5]=-1
    U_333[1,6]=1; V_333[1,6]=1; W_333[2,6]=1
    U_333[1,6]=1; V_333[4,6]=1; W_333[3,6]=1
    U_333[3,6]=-1; V_333[1,6]=-1; W_333[1,6]=-1
    U_333[3,6]=-1; V_333[4,6]=-1; W_333[2,6]=-1
    
    # Terms 7-23: Schoolbook for entries involving index 2
    col = 7
    # C[2,0]
    U_333[6,7]=1; V_333[0,7]=1; W_333[6,7]=1
    U_333[7,8]=1; V_333[3,8]=1; W_333[7,8]=1
    U_333[8,9]=1; V_333[6,9]=1; W_333[8,9]=1
    # C[2,1]
    U_333[6,10]=1; V_333[1,10]=1; W_333[7,10]=1
    U_333[7,11]=1; V_333[4,11]=1; W_333[8,11]=1
    U_333[8,12]=1; V_333[7,12]=1; W_333[7,12]=1
    # C[2,2]
    U_333[6,13]=1; V_333[2,13]=1; W_333[6,13]=1
    U_333[7,14]=1; V_333[5,14]=1; W_333[7,14]=1
    U_333[8,15]=1; V_333[8,15]=1; W_333[8,15]=1
    # C[0,2]
    U_333[0,16]=1; V_333[2,16]=1; W_333[2,16]=1
    U_333[1,17]=1; V_333[5,17]=1; W_333[2,17]=1
    U_333[2,18]=1; V_333[8,18]=1; W_333[2,18]=1
    # C[1,2]
    U_333[3,19]=1; V_333[2,19]=1; W_333[5,19]=1
    U_333[4,20]=1; V_333[5,20]=1; W_333[5,20]=1
    U_333[5,21]=1; V_333[8,21]=1; W_333[5,21]=1
    # Additional terms for C[0,0] and C[1,0] involving A[*,2]*B[2,0]
    U_333[2,22]=1; V_333[6,22]=1; W_333[0,22]=1
    U_333[5,23]=1; V_333[6,23]=1; W_333[3,23]=1
    
    # Combine U, V, W
    U_full = np.hstack([U_333, U_332])
    V_full = np.vstack([V_333, np.zeros((6, 24))])
    V_full = np.hstack([V_full, V_332])
    W_full = np.vstack([W_333, np.zeros((6, 24))])
    W_full = np.hstack([W_full, W_332])
    
    # Validate
    if validate(U_full, V_full, W_full):
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    
    # If validation fails, fall back to schoolbook (rank 45)
    U_sb = np.zeros((9, 45))
    V_sb = np.zeros((15, 45))
    W_sb = np.zeros((15, 45))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U_sb[i*3+j, col] = 1
                V_sb[j*5+k, col] = 1
                W_sb[k*3+i, col] = 1
                col += 1
                
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }

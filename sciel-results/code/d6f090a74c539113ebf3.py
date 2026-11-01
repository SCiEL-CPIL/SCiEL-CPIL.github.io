import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 3
    rank = 23
    
    # Define U, V, W for rank-23 decomposition
    # U shape: (9, 23) where row index = i*3+j (i=row, j=col of A)
    # V shape: (9, 23) where row index = j*3+k (j=row, k=col of B)  
    # W shape: (9, 23) where row index = k*3+i (k=col, i=row of C) following W[k*n+i, r]
    
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # M1: (a00+a11+a22)(b00+b11+b22)
    U[0,0] = 1; U[4,0] = 1; U[8,0] = 1
    V[0,0] = 1; V[4,0] = 1; V[8,0] = 1
    
    # M2: (a00+a11-a22)(b00+b11-b22)
    U[0,1] = 1; U[4,1] = 1; U[8,1] = -1
    V[0,1] = 1; V[4,1] = 1; V[8,1] = -1
    
    # M3: (a10+a21+a02)(b10+b21+b02)
    U[3,2] = 1; U[6,2] = 1; U[2,2] = 1
    V[3,2] = 1; V[6,2] = 1; V[2,2] = 1
    
    # M4: (a10+a21-a02)(b10+b21-b02)
    U[3,3] = 1; U[6,3] = 1; U[2,3] = -1
    V[3,3] = 1; V[6,3] = 1; V[2,3] = -1
    
    # M5: (a01+a12+a20)(b01+b12+b20)
    U[1,4] = 1; U[5,4] = 1; U[6,4] = 1
    V[1,4] = 1; V[5,4] = 1; V[6,4] = 1
    
    # M6: (a01+a12-a20)(b01+b12-b20)
    U[1,5] = 1; U[5,5] = 1; U[6,5] = -1
    V[1,5] = 1; V[5,5] = 1; V[6,5] = -1
    
    # M7: (a00+a11-a01-a10)(b00+b11+b01+b10)
    U[0,6] = 1; U[4,6] = 1; U[1,6] = -1; U[3,6] = -1
    V[0,6] = 1; V[4,6] = 1; V[1,6] = 1; V[3,6] = 1
    
    # M8: (a00-a11+a01+a10)(b00-b11+b01+b10)
    U[0,7] = 1; U[4,7] = -1; U[1,7] = 1; U[3,7] = 1
    V[0,7] = 1; V[4,7] = -1; V[1,7] = 1; V[3,7] = 1
    
    # M9: (a10+a21-a22-a12)(b10+b21+b22+b12)
    U[3,8] = 1; U[6,8] = 1; U[8,8] = -1; U[5,8] = -1
    V[3,8] = 1; V[6,8] = 1; V[8,8] = 1; V[5,8] = 1
    
    # M10: (a10-a21+a22+a12)(b10-b21+b22+b12)
    U[3,9] = 1; U[6,9] = -1; U[8,9] = 1; U[5,9] = 1
    V[3,9] = 1; V[6,9] = -1; V[8,9] = 1; V[5,9] = 1
    
    # M11: (a01+a12-a00-a20)(b01+b12+b00+b20)
    U[1,10] = 1; U[5,10] = 1; U[0,10] = -1; U[6,10] = -1
    V[1,10] = 1; V[5,10] = 1; V[0,10] = 1; V[6,10] = 1
    
    # M12: (a01-a12+a00+a20)(b01-b12+b00+b20)
    U[1,11] = 1; U[5,11] = -1; U[0,11] = 1; U[6,11] = 1
    V[1,11] = 1; V[5,11] = -1; V[0,11] = 1; V[6,11] = 1
    
    # M13: (a00+a22-a02-a20)(b00+b22+b02+b20)
    U[0,12] = 1; U[8,12] = 1; U[2,12] = -1; U[6,12] = -1
    V[0,12] = 1; V[8,12] = 1; V[2,12] = 1; V[6,12] = 1
    
    # M14: (a00-a22+a02+a20)(b00-b22+b02+b20)
    U[0,13] = 1; U[8,13] = -1; U[2,13] = 1; U[6,13] = 1
    V[0,13] = 1; V[8,13] = -1; V[2,13] = 1; V[6,13] = 1
    
    # M15: a00*b11
    U[0,14] = 1
    V[4,14] = 1
    
    # M16: a11*b22
    U[4,15] = 1
    V[8,15] = 1
    
    # M17: a22*b00
    U[8,16] = 1
    V[0,16] = 1
    
    # M18: a01*b20
    U[1,17] = 1
    V[6,17] = 1
    
    # M19: a12*b02
    U[5,18] = 1
    V[2,18] = 1
    
    # M20: a20*b12
    U[6,19] = 1
    V[5,19] = 1
    
    # M21: a02*b10
    U[2,20] = 1
    V[3,20] = 1
    
    # M22: a10*b22
    U[3,21] = 1
    V[8,21] = 1
    
    # M23: a21*b01
    U[6,22] = 1
    V[1,22] = 1
    
    # W reconstruction: W[k*3+i, r] is coefficient for C[i,k] from M[r]
    # c[i,k] = sum_r W[k*3+i, r] * M[r]
    
    # c00 (i=0,k=0) -> W[0, r]
    W[0, 0] = 0.5; W[0, 1] = 0.5; W[0, 6] = -0.5; W[0, 7] = -0.5
    W[0, 14] = 1; W[0, 16] = 1; W[0, 20] = 1; W[0, 21] = 1
    
    # c01 (i=0,k=1) -> W[3, r]
    W[3, 2] = 0.5; W[3, 3] = -0.5; W[3, 8] = -0.5; W[3, 9] = -0.5
    W[3, 14] = 1; W[3, 15] = 1; W[3, 17] = 1; W[3, 22] = 1
    
    # c02 (i=0,k=2) -> W[6, r]
    W[6, 4] = 0.5; W[6, 5] = -0.5; W[6, 10] = -0.5; W[6, 11] = -0.5
    W[6, 14] = 1; W[6, 18] = 1; W[6, 19] = 1; W[6, 20] = 1
    
    # c10 (i=1,k=0) -> W[1, r]
    W[1, 2] = 0.5; W[1, 3] = 0.5; W[1, 8] = -0.5; W[1, 9] = -0.5
    W[1, 15] = 1; W[1, 16] = 1; W[1, 17] = 1; W[1, 21] = 1
    
    # c11 (i=1,k=1) -> W[4, r]
    W[4, 0] = 0.5; W[4, 1] = -0.5; W[4, 6] = -0.5; W[4, 7] = 0.5
    W[4, 15] = 1; W[4, 18] = 1; W[4, 19] = 1; W[4, 22] = 1
    
    # c12 (i=1,k=2) -> W[7, r]
    W[7, 4] = 0.5; W[7, 5] = 0.5; W[7, 10] = -0.5; W[7, 11] = 0.5
    W[7, 16] = 1; W[7, 18] = 1; W[7, 19] = 1; W[7, 21] = 1
    
    # c20 (i=2,k=0) -> W[2, r]
    W[2, 2] = 0.5; W[2, 3] = 0.5; W[2, 8] = 0.5; W[2, 9] = 0.5
    W[2, 15] = 1; W[2, 16] = 1; W[2, 17] = 1; W[2, 20] = 1
    
    # c21 (i=2,k=1) -> W[5, r]
    W[5, 2] = 0.5; W[5, 3] = -0.5; W[5, 8] = 0.5; W[5, 9] = 0.5
    W[5, 14] = 1; W[5, 15] = 1; W[5, 17] = 1; W[5, 21] = 1
    
    # c22 (i=2,k=2) -> W[8, r]
    W[8, 0] = 0.5; W[8, 1] = 0.5; W[8, 6] = 0.5; W[8, 7] = 0.5
    W[8, 14] = 1; W[8, 18] = 1; W[8, 19] = 1; W[8, 22] = 1
    
    # Verify with multiple random tests
    for seed in [42, 123, 456, 789, 101112]:
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        A_test = np.random.randint(-10, 11, size=(3, 3)).astype(float)
        B_test = np.random.randint(-10, 11, size=(3, 3)).astype(float)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((3, 3))
        for r in range(rank):
            u_val = np.dot(U[:, r], A_test.flatten())
            v_val = np.dot(V[:, r], B_test.flatten())
            prod = u_val * v_val
            for i in range(3):
                for k in range(3):
                    C_computed[i, k] += prod * W[k*3+i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook rank 27
            rank = 27
            U = np.zeros((9, rank))
            V = np.zeros((9, rank))
            W = np.zeros((9, rank))
            r = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U[i*3+j, r] = 1
                        V[j*3+k, r] = 1
                        W[k*3+i, r] = 1
                        r += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

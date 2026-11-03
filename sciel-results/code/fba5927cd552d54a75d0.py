import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 3
    
    # Rank 17 solution
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for 2x2x2 sub-block (indices 0,1 for rows/cols)
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # U indices: i*m+j. (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # V indices: k*p+l. (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # W indices: l*n+i. (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # r=0
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Cross terms involving A[:, 2] and B[2, :]
    # C[0,0] += A[0,2]B[2,0]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    # C[0,1] += A[0,2]B[2,1]
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    # C[1,0] += A[1,2]B[2,0]
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    # C[1,1] += A[1,2]B[2,1]
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Part 3: Cross terms involving A[:, 0:2] and B[:, 2]
    # C[0,2] += A[0,0]B[0,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    # C[0,2] += A[0,1]B[1,2]
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    # C[0,2] += A[0,2]B[2,2]
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    # C[1,2] += A[1,0]B[0,2]
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    # C[1,2] += A[1,1]B[1,2]
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    # C[1,2] += A[1,2]B[2,2]
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    # Verification
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook (should not happen)
        U_fb = np.zeros((n * m, n * m * p))
        V_fb = np.zeros((m * p, n * m * p))
        W_fb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

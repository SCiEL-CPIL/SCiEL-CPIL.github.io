"""Matrix multiplication 3x4x5 with Strassen-based low-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 5

    def strassen_2x2_terms():
        """Return list of 7 (U, V, W) for 2x2x2 Strassen.
        U[i,j], V[j,k], W[k,i] (W indexed by k*2+i for 2x2).
        """
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        u1 = [[1, 0], [0, 1]]
        v1 = [[1, 0], [0, 1]]
        w1 = [[1, 0], [0, 1]]  # W[k,i]: C[0,0] and C[1,1]
        terms.append((u1, v1, w1))

        # M2 = (A10+A11)(B00) -> C10 += M2
        u2 = [[0, 0], [1, 1]]
        v2 = [[1, 0], [0, 0]]
        w2 = [[0, 0], [1, 0]]  # C[1,0]: k=1,i=0
        terms.append((u2, v2, w2))

        # M3 = A00(B01-B11) -> C01 += M3
        u3 = [[1, 0], [0, 0]]
        v3 = [[0, 1], [0, -1]]
        w3 = [[0, 0], [0, 1]]  # C[0,1]: k=0,i=1
        terms.append((u3, v3, w3))

        # M4 = A11(B10-B00) -> C10 += M4, C00 += M4
        u4 = [[0, 0], [0, 1]]
        v4 = [[-1, 0], [1, 0]]
        w4 = [[1, 0], [1, 0]]  # C[0,0]: k=0,i=0; C[1,0]: k=1,i=0
        terms.append((u4, v4, w4))

        # M5 = (A00+A01)(B11) -> C01 += M5, C00 -= M5
        u5 = [[1, 1], [0, 0]]
        v5 = [[0, 0], [0, 1]]
        w5 = [[-1, 1], [0, 0]]  # C[0,0]: -M5; C[0,1]: +M5
        terms.append((u5, v5, w5))

        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        u6 = [[-1, 0], [1, 0]]
        v6 = [[1, 1], [0, 0]]
        w6 = [[0, 0], [0, 1]]  # C[1,1]: k=1,i=1
        terms.append((u6, v6, w6))

        # M7 = (A01-A11)(B10+B11) -> C11 += M7, C00 += M7
        u7 = [[0, 1], [0, -1]]
        v7 = [[0, 0], [1, 1]]
        w7 = [[1, 0], [0, 1]]  # C[0,0]: +M7; C[1,1]: +M7
        terms.append((u7, v7, w7))

        return terms

    def expand_2x2_to_4x4(terms_2x2):
        """Expand 7 terms of 2x2 Strassen into 49 terms for 4x4x4."""
        all_terms = []
        for (U2, V2, W2) in terms_2x2:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        cu = U2[i][j]
                        cv = V2[j][k]
                        cw = W2[k][i]
                        if cu == 0 or cv == 0 or cw == 0:
                            continue
                        for (su, sv, sw) in strassen_2x2_terms():
                            U4 = [[0]*4 for _ in range(4)]
                            V4 = [[0]*4 for _ in range(4)]
                            W4 = [[0]*4 for _ in range(4)]
                            
                            for di in range(2):
                                for dj in range(2):
                                    U4[i*2+di][j*2+dj] = cu * su[di][dj]
                            
                            for dj in range(2):
                                for dk in range(2):
                                    V4[j*2+dj][k*2+dk] = cv * sv[dj][dk]
                            
                            for dk in range(2):
                                for di in range(2):
                                    W4[k*2+dk][i*2+di] = cw * sw[dk][di]
                            
                            all_terms.append((U4, V4, W4))
        
        return all_terms

    # Get 49 terms for 4x4x4
    terms_2x2 = strassen_2x2_terms()
    terms_4x4 = expand_2x2_to_4x4(terms_2x2)
    
    if time.time() > deadline - 5:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    total_rank = len(terms_4x4) + 3  # 49 + 3 = 52
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 52
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 52
    W = [[0] * total_rank for _ in range(p * n)]  # 15 x 52
    
    # Fill in the 49 terms from 4x4x4, restricted to 3x4x4
    for r, (U4, V4, W4) in enumerate(terms_4x4):
        for i in range(min(4, n)):
            for j in range(min(4, m)):
                U[i*m + j][r] = U4[i][j]
        
        for j in range(min(4, m)):
            for k in range(min(4, p)):
                V[j*p + k][r] = V4[j][k]
        
        for k in range(min(4, p)):
            for i in range(min(4, n)):
                W[k*n + i][r] = W4[k][i]
    
    # Fill in the 3 terms for 3x4x1 (column 4 of C)
    for j in range(m):
        r = 49 + j
        for i in range(n):
            U[i*m + j][r] = 1
        V[j*p + 4][r] = 1
        for i in range(n):
            W[4*n + i][r] = 1
    
    # Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    if time.time() < deadline - 2:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

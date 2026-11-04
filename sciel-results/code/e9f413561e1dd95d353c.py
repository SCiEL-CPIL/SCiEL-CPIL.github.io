import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4, 7), (4, 7), (4, 7).
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 4
    start_time = time.time()
    deadline = start_time + 140
    
    # Construct rank-49 decomposition using Kronecker product of Strassen
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-compute non-zero entries for each rank term of the 2x2 Strassen decomposition
    # to speed up the Kronecker product construction.
    U2_nz = []
    V2_nz = []
    W2_nz = []
    for r in range(7):
        u_list = [(i, U2[i][r]) for i in range(4) if U2[i][r] != 0]
        v_list = [(j, V2[j][r]) for j in range(4) if V2[j][r] != 0]
        w_list = [(k, W2[k][r]) for k in range(4) if W2[k][r] != 0]
        U2_nz.append(u_list)
        V2_nz.append(v_list)
        W2_nz.append(w_list)

    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # Iterate over non-zero elements of U2, V2, W2 for terms r1 and r2
            for i1_idx, u1_val in U2_nz[r1]:
                for i2_idx, u2_val in U2_nz[r2]:
                    i1 = i1_idx // 2
                    k1 = i1_idx % 2
                    i2 = i2_idx // 2
                    k2 = i2_idx % 2
                    
                    i = 2 * i1 + i2
                    k = 2 * k1 + k2
                    u_total = u1_val * u2_val
                    U[i * m + k][r] += u_total
            
            for j1_idx, v1_val in V2_nz[r1]:
                for j2_idx, v2_val in V2_nz[r2]:
                    k1 = j1_idx // 2
                    j1 = j1_idx % 2
                    k2 = j2_idx // 2
                    j2 = j2_idx % 2
                    
                    k = 2 * k1 + k2
                    j = 2 * j1 + j2
                    v_total = v1_val * v2_val
                    V[k * p + j][r] += v_total
            
            for w1_idx, w1_val in W2_nz[r1]:
                for w2_idx, w2_val in W2_nz[r2]:
                    j1 = w1_idx // 2
                    i1 = w1_idx % 2
                    j2 = w2_idx // 2
                    i2 = w2_idx % 2
                    
                    j = 2 * j1 + j2
                    i = 2 * i1 + i2
                    w_total = w1_val * w2_val
                    W[j * n + i][r] += w_total

    return {"U": U, "V": V, "W": W}

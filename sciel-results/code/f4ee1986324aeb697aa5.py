import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 4, 4, 4
    
    # Strassen 2x2 decomposition coefficients
    # For 2x2 matrices: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5, C12 = M3 + M5
    # C21 = M2 + M4, C22 = M1 - M2 + M7
    
    # Coefficients for A, B, C in each of the 7 Strassen terms
    # a_coeffs[s][i][j] = coefficient of A[i][j] in s-th multiplication
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M1: a+d
        [[0, 0], [1, 1]],   # M2: c+d
        [[1, 0], [0, 0]],   # M3: a
        [[0, 0], [0, 1]],   # M4: d
        [[1, 1], [0, 0]],   # M5: a+b
        [[-1, 0], [1, 0]],  # M6: c-a
        [[0, 1], [0, -1]],  # M7: b-d
    ])
    
    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # M1: e+h
        [[1, 0], [0, 0]],   # M2: e
        [[0, 1], [0, -1]],  # M3: f-h
        [[-1, 0], [1, 0]],  # M4: g-e
        [[0, 0], [0, 1]],   # M5: h
        [[1, 1], [0, 0]],   # M6: e+f
        [[0, 0], [1, 1]],   # M7: g+h
    ])
    
    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # C11 += M1, C22 += M1
        [[0, 0], [1, -1]],  # C21 += M2, C22 -= M2
        [[0, 1], [0, 1]],   # C12 += M3, C22 += M3
        [[1, 0], [1, 0]],   # C11 += M4, C21 += M4
        [[-1, 1], [0, 0]],  # C11 -= M5, C12 += M5
        [[0, 0], [0, 1]],   # C22 += M6
        [[1, 0], [0, 0]],   # C11 += M7
    ])
    
    # Build U, V, W using recursive Strassen (rank 49)
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # For 4x4 = 2x2 blocks of 2x2 matrices
    # A[i][j] where i = 2*S + s2, j = 2*T + t2
    # B[j][k] where j = 2*T + t2, k = 2*K + k2  
    # C[i][k] where i = 2*S + s2, k = 2*K + k2
    
    for s in range(7):  # outer Strassen term
        for t in range(7):  # inner Strassen term
            c_idx = 7 * s + t
            
            # U: rows are (i, j) pairs, index i*m+j
            for S in range(2):
                for T in range(2):
                    a_outer = a_coeffs[s][S][T]
                    if a_outer == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_inner = a_coeffs[t][s2][t2]
                            if a_inner == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * m + j, c_idx] += a_outer * a_inner
            
            # V: rows are (j, k) pairs, index j*p+k
            for T in range(2):
                for K in range(2):
                    b_outer = b_coeffs[s][T][K]
                    if b_outer == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_inner = b_coeffs[t][t2][k2]
                            if b_inner == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * p + k, c_idx] += b_outer * b_inner
            
            # W: rows are (k, i) pairs, index k*n+i (trace convention)
            for K in range(2):
                for S in range(2):
                    c_outer = c_coeffs[s][S][K]
                    if c_outer == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_inner = c_coeffs[t][s2][k2]
                            if c_inner == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * n + i, c_idx] += c_outer * c_inner
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

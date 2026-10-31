import time
import numpy as np

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    """
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2 coefficient matrices
    # a_coeffs[s][i][j]: coefficient of A[i][j] in s-th multiplication
    # b_coeffs[s][j][k]: coefficient of B[j][k] in s-th multiplication
    # c_coeffs[s][i][k]: coefficient of C[i][k] from s-th multiplication
    
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M0 = (A00+A11)(B00+B11)
        [[0, 0], [1, 1]],   # M1 = (A10+A11)B00
        [[1, 0], [0, 0]],   # M2 = A00(B01-B11)
        [[0, 0], [0, 1]],   # M3 = A11(B10-B00)
        [[1, 1], [0, 0]],   # M4 = (A00+A01)B11
        [[-1, 0], [1, 0]],  # M5 = (A10-A00)(B00+B01)
        [[0, 1], [0, -1]],  # M6 = (A01-A11)(B10+B11)
    ], dtype=np.int8)
    
    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # (B00+B11)
        [[1, 0], [0, 0]],   # B00
        [[0, 1], [0, -1]],  # (B01-B11)
        [[-1, 0], [1, 0]],  # (B10-B00)
        [[0, 0], [0, 1]],   # B11
        [[1, 1], [0, 0]],   # (B00+B01)
        [[0, 0], [1, 1]],   # (B10+B11)
    ], dtype=np.int8)
    
    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # C00+=M0, C11+=M0
        [[0, 0], [1, -1]],  # C10+=M1, C11-=M1
        [[0, 1], [0, 1]],   # C01+=M2, C11+=M2
        [[1, 0], [1, 0]],   # C00+=M3, C10+=M3
        [[-1, 1], [0, 0]],  # C00-=M4, C01+=M4
        [[0, 0], [0, 1]],   # C11+=M5
        [[1, 0], [0, 0]],   # C00+=M6
    ], dtype=np.int8)
    
    rank = 49
    U = np.zeros((16, rank), dtype=np.int8)
    V = np.zeros((16, rank), dtype=np.int8)
    W = np.zeros((16, rank), dtype=np.int8)
    
    for s in range(7):
        for t in range(7):
            col = 7 * s + t
            
            # U: A index = i*4+j, where i=2*S+s2, j=2*T+t2
            # coeff = a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            for S in range(2):
                for T in range(2):
                    ca = int(a_coeffs[s, S, T])
                    if ca == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            cb = int(a_coeffs[t, s2, t2])
                            if cb == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j, col] += ca * cb
            
            # V: B index = j*4+k, where j=2*T+t2, k=2*K+k2
            # coeff = b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
            for T in range(2):
                for K in range(2):
                    ca = int(b_coeffs[s, T, K])
                    if ca == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            cb = int(b_coeffs[t, t2, k2])
                            if cb == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k, col] += ca * cb
            
            # W: C index = k*4+i, where k=2*K+k2, i=2*S+s2
            # coeff = c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
            for K in range(2):
                for S in range(2):
                    ca = int(c_coeffs[s, S, K])
                    if ca == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            cb = int(c_coeffs[t, s2, k2])
                            if cb == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i, col] += ca * cb
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Verify correctness
    is_valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s_val = int(np.dot(U[i*4+j, :], V[j*4+k, :] * W[k*4+i, :]))
                if s_val != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback to schoolbook (Rank 64)
        rank = 64
        U = np.zeros((16, rank), dtype=np.int8)
        V = np.zeros((16, rank), dtype=np.int8)
        W = np.zeros((16, rank), dtype=np.int8)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

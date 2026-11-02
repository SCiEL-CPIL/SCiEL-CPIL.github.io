import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 2, 12
    
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen decomposition for 2x2x2 applied to each pair of columns
    # A is 2x2, B_pair is 2x2 (cols 2t, 2t+1), C_pair is 2x2
    # T[i,j,k] = 1
    
    for t in range(num_pairs):
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        for s in range(7):
            c = base + s
            
            if s == 0:
                # M1 = (A00+A11)*(B00+B11)
                U[0, c] += 1   # A[0,0]
                U[3, c] += 1   # A[1,1]
                V[0*p + k0, c] += 1   # B[0, k0]
                V[1*p + k1, c] += 1   # B[1, k1]
                W[k0*n + 0, c] += 1   # C[0, k0]
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 1:
                # M2 = (A10+A11)*B00
                U[2, c] += 1   # A[1,0]
                U[3, c] += 1   # A[1,1]
                V[0*p + k0, c] += 1   # B[0, k0]
                W[k0*n + 1, c] += 1   # C[1, k0]
                W[k1*n + 1, c] += -1  # C[1, k1]
                
            elif s == 2:
                # M3 = A00*(B01-B11)
                U[0, c] += 1   # A[0,0]
                V[0*p + k1, c] += 1    # B[0, k1]
                V[1*p + k1, c] += -1   # B[1, k1]
                W[k1*n + 0, c] += 1   # C[0, k1]
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 3:
                # M4 = A11*(B10-B00)
                U[3, c] += 1   # A[1,1]
                V[1*p + k0, c] += 1    # B[1, k0]
                V[0*p + k0, c] += -1   # B[0, k0]
                W[k0*n + 0, c] += 1   # C[0, k0]
                W[k0*n + 1, c] += 1   # C[1, k0]
                
            elif s == 4:
                # M5 = (A00+A01)*B11
                U[0, c] += 1   # A[0,0]
                U[1, c] += 1   # A[0,1]
                V[1*p + k1, c] += 1   # B[1, k1]
                W[k0*n + 0, c] += -1  # C[0, k0]
                W[k1*n + 0, c] += 1   # C[0, k1]
                
            elif s == 5:
                # M6 = (A10-A00)*(B00+B01)
                U[2, c] += 1   # A[1,0]
                U[0, c] += -1  # A[0,0]
                V[0*p + k0, c] += 1    # B[0, k0]
                V[0*p + k1, c] += 1    # B[0, k1]
                W[k1*n + 1, c] += 1   # C[1, k1]
                
            elif s == 6:
                # M7 = (A01-A11)*(B10+B11)
                U[1, c] += 1   # A[0,1]
                U[3, c] += -1  # A[1,1]
                V[1*p + k0, c] += 1    # B[1, k0]
                V[1*p + k1, c] += 1    # B[1, k1]
                W[k0*n + 0, c] += 1   # C[0, k0]
    
    # Quick verification (optional but safe)
    # U_arr = np.array(U)
    # V_arr = np.array(V)
    # W_arr = np.array(W)
    # max_err = 0.0
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    #             if abs(val - 1.0) > max_err:
    #                 max_err = abs(val - 1.0)
    # if max_err > 1e-9:
    #     # Fallback to schoolbook if verification fails (should not happen)
    #     rank = 48
    #     U = np.zeros((n * m, rank))
    #     V = np.zeros((m * p, rank))
    #     W = np.zeros((p * n, rank))
    #     r = 0
    #     for k in range(p):
    #         for i in range(n):
    #             for j in range(m):
    #                 U[i * m + j, r] = 1
    #                 V[j * p + k, r] = 1
    #                 W[k * n + i, r] = 1
    #                 r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition_fast(U, V, W, n, m, p):
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U_r[a,b,:] * V_r[c,d,:] * W_r[k,i,:])
                            if abs(s - target) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Block decomposition: 6 = 3 * 2
    # C[i, k] = sum_j A[i,j] * B[j,k] where k = 3*j1 + j2
    # For each j1 in {0,1}, fix the block of B columns
    # Schoolbook: for each (i,j), one term that contributes to all 3 outputs in that block
    # Total rank = 2 (blocks) * 2 (rows of A) * 3 (cols of A) = 12
    
    R = 12
    U = np.zeros((6, R))   # n*m = 2*3 = 6 rows
    V = np.zeros((18, R))  # m*p = 3*6 = 18 rows
    W = np.zeros((12, R))  # p*n = 6*2 = 12 rows
    
    for j1 in range(2):
        base = j1 * 6
        for i in range(2):
            for j in range(3):
                r = base + i * 3 + j
                # U index: a*3 + b where a=i, b=j
                U[i*3 + j, r] = 1
                # V index: c*6 + d where c=j, d=3*j1+j2
                for j2 in range(3):
                    d = 3 * j1 + j2
                    V[j*6 + d, r] = 1
                    # W index: k*2 + i_out where k=d, i_out=i
                    W[d*2 + i, r] = 1
    
    if time.time() < deadline:
        if check_decomposition_fast(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook rank 36
    U_school = np.zeros((6, 36))
    V_school = np.zeros((18, 36))
    W_school = np.zeros((12, 36))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U_school[i * 3 + j, col] = 1
                V_school[j * 6 + k, col] = 1
                W_school[k * 2 + i, col] = 1
                col += 1
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

if __name__ == "__main__":
    construct()

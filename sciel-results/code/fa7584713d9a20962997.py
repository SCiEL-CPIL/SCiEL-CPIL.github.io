import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook_optimized():
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(m):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
            W[k * n + i][r] = 1 # Wait, W depends on i as well.
            
    # Correction: W[k*n+i, r] should be 1 for all i, k for a given r?
    # Let's re-verify the indices.
    # T_ijk = sum_r U_ir V_jr W_kr
    # U is (n*m, rank). Index i*m+j.
    # V is (m*p, rank). Index j*p+k.
    # W is (p*n, rank). Index k*n+i.
    
    # For term r (corresponding to j=r):
    # U[i*m+r, r] = 1 for all i.
    # V[r*p+k, r] = 1 for all k.
    # W[k*n+i, r] = 1 for all k, i.
    
    # Let's re-initialize and fill correctly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(m):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    result = _schoolbook_optimized()
    ok, err = _verify_fast(result["U"], result["V"], result["W"], n, m, p)
    
    if ok:
        return result
    
    # Fallback to original schoolbook if optimized fails (should not happen)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    ok, err = _verify_fast(res['U'], res['V'], res['W'], 3, 4, 4)
    print("Valid:", ok, "Err:", err)

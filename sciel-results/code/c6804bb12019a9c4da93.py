import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Strassen-based rank-35 solution
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = 7 * (p // 2)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U[idx, col] = a_vec[idx]
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            W[2 * k * n + 0, col] = c_vec[0]
            W[2 * k * n + 1, col] = c_vec[1]
            W[(2 * k + 1) * n + 0, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            col += 1
    
    # Verify the solution
    def verify(U, V, W, n, m, p, tol=1e-6):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i*m + k, k*p + j, j*n + i] = 1.0
        result = np.zeros((n*m, m*p, p*n))
        for k in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:, k], V[:, k], W[:, k])
        return np.allclose(result, T, atol=tol)
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_sb[i*m + k, col] = 1
                    V_sb[k*p + j, col] = 1
                    W_sb[j*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

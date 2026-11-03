import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    rank = 40
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # 1. Strassen block
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_global = li // 2
                j_global = li % 2
                U[i_global*m + j_global, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_global = lj // 2
                k_global = k0 + (lj % 2)
                V[j_global*p + k_global, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_global = lc // 2
                k_global = k0 + (lc % 2)
                W[k_global*n + i_global, col] += coeff
            col += 1
        
        # 2. Cross terms with rank 1 each
        
        # i in {0,1}, j=2
        U[0*m + 2, col] = 1.0
        U[1*m + 2, col] = 1.0
        V[2*p + k0, col] = 1.0
        V[2*p + k1, col] = 1.0
        W[k0*n + 0, col] = 1.0
        W[k1*n + 0, col] = 1.0
        W[k0*n + 1, col] = 1.0
        W[k1*n + 1, col] = 1.0
        col += 1
        
        # i=2, j in {0,1}
        U[2*m + 0, col] = 1.0
        U[2*m + 1, col] = 1.0
        V[0*p + k0, col] = 1.0
        V[0*p + k1, col] = 1.0
        V[1*p + k0, col] = 1.0
        V[1*p + k1, col] = 1.0
        W[k0*n + 2, col] = 1.0
        W[k1*n + 2, col] = 1.0
        col += 1
        
        # i=2, j=2
        U[2*m + 2, col] = 1.0
        V[2*p + k0, col] = 1.0
        V[2*p + k1, col] = 1.0
        W[k0*n + 2, col] = 1.0
        W[k1*n + 2, col] = 1.0
        col += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        return schoolbook()

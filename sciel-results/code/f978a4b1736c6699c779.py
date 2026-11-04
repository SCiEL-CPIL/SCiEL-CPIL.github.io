import time
import numpy as np

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 120
    
    # Rank 27 schoolbook (fallback)
    def get_rank_27():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition
    def verify(U_mat, V_mat, W_mat):
        Ua = np.array(U_mat, dtype=float)
        Va = np.array(V_mat, dtype=float)
        Wa = np.array(W_mat, dtype=float)
        
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n*m):
            ra, ca = divmod(i, m)
            for j in range(m*p):
                rb, cb = divmod(j, p)
                if ca == rb:
                    k = ra * n + cb
                    T_exp[i, j, k] = 1.0
        
        T_comp = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        return np.allclose(T_comp, T_exp, atol=1e-10)
    
    # Return rank 27 (guaranteed correct)
    return get_rank_27()

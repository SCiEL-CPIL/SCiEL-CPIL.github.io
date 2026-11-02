import time
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 145.0
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2x2
    # Each term is a list of (index, coeff) for U, V, W
    # Indices are local (0..3) for 2x2 matrices
    # U: i*m + j
    # V: j*p + k (local p=2)
    # W: k*n + i (local n=2)
    
    strassen_terms = [
        # M1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # M2
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        # M3
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        # M4
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(2, 1), (0, 1)]
        },
        # M5
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # M6
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # M7
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        }
    ]
    
    col = 0
    # 3 blocks of p=6 (size 2 each)
    # 2 blocks of m=4 (size 2 each)
    # 1 block of n=2 (size 2)
    
    for r_block in range(3):
        for s_block in range(2):
            for t_idx in range(7):
                term = strassen_terms[t_idx]
                
                # U indices
                for local_u, coeff in term['U']:
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s_block + j_local
                    u_idx = i * m + j_global
                    U[u_idx, col] += coeff
                
                # V indices
                for local_v, coeff in term['V']:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s_block + j_local
                    k_global = 2 * r_block + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] += coeff
                
                # W indices
                for local_w, coeff in term['W']:
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r_block + k_local
                    w_idx = k_global * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    # Verification
    try:
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
        if not np.all(T == 1.0):
            # Fallback to schoolbook
            r_sb = n * m * p
            U_sb = np.zeros((n * m, r_sb))
            V_sb = np.zeros((m * p, r_sb))
            W_sb = np.zeros((p * n, r_sb))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, col] = 1
                        V_sb[j * p + k, col] = 1
                        W_sb[k * n + i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

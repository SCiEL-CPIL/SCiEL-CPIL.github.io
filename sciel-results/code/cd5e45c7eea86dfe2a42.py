import numpy as np

def construct():
    n, m, p = 3, 4, 5
    target_rank = 56
    
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
    ]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    current_rank = 0
    covered = set()
    
    for I, J, K in blocks:
        for u, v, w in strassen_terms:
            r = current_rank
            current_rank += 1
            
            for li in range(2):
                for lj in range(2):
                    u_val = u[li * 2 + lj]
                    if u_val != 0:
                        gi = I[li]
                        gj = J[lj]
                        U[gi * m + gj, r] = u_val
                        
            for lj in range(2):
                for lk in range(2):
                    v_val = v[lj * 2 + lk]
                    if v_val != 0:
                        gj = J[lj]
                        gk = K[lk]
                        V[gj * p + gk, r] = v_val
                        
            for lk in range(2):
                for li in range(2):
                    w_val = w[lk * 2 + li]
                    if w_val != 0:
                        gk = K[lk]
                        gi = I[li]
                        W[gk * n + gi, r] = w_val
                        
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        covered.add((I[li], J[lj], K[lk]))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    r = current_rank
                    current_rank += 1
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

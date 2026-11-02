import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    time_limit = 110.0
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k * n + i] = 1.0

    def verify(U, V, W):
        T_recon = np.zeros((dim_u, dim_v, dim_w))
        rank = U.shape[1]
        for r in range(rank):
            T_recon += np.einsum('i,j,l->ijl', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_recon, T, atol=1e-9)

    def round_to_half(x):
        return np.round(x * 2) / 2

    best_rank = 100
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook
    U_school = np.zeros((dim_u, best_rank))
    V_school = np.zeros((dim_v, best_rank))
    W_school = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Search
    for r in range(99, 49, -1):
        if time.time() - start_time > time_limit:
            break
            
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            U_rand = np.random.randint(-1, 2, size=(dim_u, r))
            V_rand = np.random.randint(-1, 2, size=(dim_v, r))
            
            W_sol = np.zeros((dim_w, r))
            valid = True
            for l in range(dim_w):
                M_l = np.einsum('ir,jr->ijr', U_rand, V_rand).reshape(-1, r)
                b_l = T[:, :, l].reshape(-1)
                try:
                    W_l, _, _, _ = np.linalg.lstsq(M_l, b_l, rcond=None)
                    W_sol[l, :] = W_l
                except:
                    valid = False
                    break
            
            if not valid:
                continue
                
            W_rounded = round_to_half(W_sol)
            
            if verify(U_rand, V_rand, W_rounded):
                best_rank = r
                best_U = U_rand
                best_V = V_rand
                best_W = W_rounded
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}

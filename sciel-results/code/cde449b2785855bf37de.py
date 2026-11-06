import numpy as np
import time
import math
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit
    
    N = 24
    # Search grid for local search
    M_search = 4096
    z_search = np.exp(2j * np.pi * np.arange(M_search) / M_search)
    # Precompute Z matrix: Z[j, k] = z_search[j] ** k
    Z = np.power(z_search[:, None], np.arange(N)[None, :])
    
    def get_max_norm(c, Z_mat):
        vals = Z_mat @ c
        return np.max(np.abs(vals))
    
    def local_search(c, Z_mat, time_limit):
        c = np.array(c, dtype=np.float64)
        vals = Z_mat @ c
        current_max = np.max(np.abs(vals))
        
        improved = True
        max_iter = 1000
        iter_count = 0
        
        while improved and iter_count < max_iter and (time.time() - start_time) < time_limit:
            improved = False
            iter_count += 1
            
            best_gain = 0
            best_k = -1
            best_new_vals = None
            
            for k in range(N):
                if c[k] == 1:
                    delta = -2 * Z_mat[:, k]
                else:
                    delta = 2 * Z_mat[:, k]
                
                new_vals = vals + delta
                new_max = np.max(np.abs(new_vals))
                
                if new_max < current_max - 1e-12:
                    gain = current_max - new_max
                    if gain > best_gain:
                        best_gain = gain
                        best_k = k
                        best_new_vals = new_vals
            
            if best_k != -1:
                c[best_k] = -c[best_k]
                vals = best_new_vals
                current_max = current_max - best_gain
                improved = True
        
        return c

    # Initial candidates
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    gRS = np.array(p[:N], dtype=np.float64)
    
    candidates = [gRS]
    
    np.random.seed(42)
    for _ in range(100):
        rand_c = np.random.choice([-1, 1], N)
        candidates.append(rand_c)
        
    for i in range(N):
        c_flip = gRS.copy()
        c_flip[i] = -c_flip[i]
        candidates.append(c_flip)

    best_c = None
    best_score = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 5.0:
            break
        c_ls = local_search(c, Z, deadline - 5.0)
        score_ls = get_max_norm(c_ls, Z)
        if score_ls < best_score:
            best_score = score_ls
            best_c = c_ls
            
    if best_c is None:
        return list(gRS.astype(int))
        
    return list(best_c.astype(int))

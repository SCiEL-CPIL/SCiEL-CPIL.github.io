import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s wall-clock kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
        
    def try_find_rank(R, time_budget=10.0):
        start_time = time.time()
        max_attempts = 10000
        values = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        
        # Precompute indices for faster C construction
        # e = i*m*p + j*p + k
        # col_base = r * pn
        # w_idx = k*n + i
        # u_idx = i*m + j
        # v_idx = j*p + k
        
        # We can precompute the mapping from (i,j,k) to (e, u_idx, v_idx, w_idx)
        # and for each r, the column offset is r * pn.
        
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = i * m * p + j * p + k
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    indices.append((e, u_idx, v_idx, w_idx))
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Generate random U and V
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # Increase sparsity slightly to allow more complex structures
            for r in range(R):
                n_u_nonzero = random.randint(2, 5)
                n_v_nonzero = random.randint(2, 5)
                
                u_indices = random.sample(range(nm), min(n_u_nonzero, nm))
                v_indices = random.sample(range(mp), min(n_v_nonzero, mp))
                
                for idx in u_indices:
                    U[idx, r] = random.choice(values)
                for idx in v_indices:
                    V[idx, r] = random.choice(values)
            
            # Build coefficient matrix C for W
            # C is (n*m*p, R * pn)
            C = np.zeros((n * m * p, R * pn))
            
            # Vectorized-ish construction
            # For each r, we need to fill columns r*pn + w_idx
            # C[e, r*pn + w_idx] = U[u_idx, r] * V[v_idx, r]
            
            # We can iterate over the indices list
            # To speed up, we can pre-compute the column indices for each r
            # col_indices_for_r = [r*pn + w_idx for (e, u_idx, v_idx, w_idx) in indices]
            # This is constant for a given R and indices list.
            
            # Let's precompute this outside the attempt loop?
            # No, R changes. But within a single try_find_rank call, R is fixed.
            
            # Precompute column offsets for each equation e
            # cols_for_r[r] = [r*pn + w_idx for ...]
            # But w_idx depends on i,k which are in indices.
            
            # Let's just use the loop, it's only 18 iterations * R (17) = ~300 ops per attempt.
            # 10000 attempts * 300 ops = 3M ops. This is fast in Python? 
            # Maybe not. 10k attempts * (18 inner loop) = 180k iterations.
            # This is very fast.
            
            for e, u_idx, v_idx, w_idx in indices:
                for r in range(R):
                    u_val = U[u_idx, r]
                    v_val = V[v_idx, r]
                    if u_val != 0.0 and v_val != 0.0:
                        C[e, r * pn + w_idx] = u_val * v_val
            
            b = np.ones(n * m * p)
            
            # Solve linear system
            try:
                w_flat, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            W = w_flat.reshape(pn, R)
            
            if verify(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
                
        return None

    # Try ranks from 17 down to 12, prioritizing 17
    # The hint says 17 is achievable.
    ranks_to_try = [17, 16, 15, 14, 13, 12]
    
    for R in ranks_to_try:
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 5.0:
            break
        # Give rank 17 more time
        if R == 17:
            time_budget = min(60.0, remaining - 5.0)
        else:
            time_budget = min(10.0, remaining - 5.0)
            
        result = try_find_rank(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
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

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
    
    def verify_full(U, V, W):
        """Full verification: check all 6*9*6=324 entries of the tensor."""
        R = U.shape[1]
        # Check all valid (i,j,k) entries give 1
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
        # Check all off-diagonal entries give 0
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    i = u // m
                    u_j = u % m
                    v_j = v // p
                    k_v = v % p
                    k_w = w // n
                    i_w = w % n
                    if i == i_w and u_j == v_j and k_v == k_w:
                        continue
                    val = 0.0
                    for r in range(R):
                        val += U[u, r] * V[v, r] * W[w, r]
                    if abs(val) > 1e-9:
                        return False
        return True

    def verify_partial(U, V, W):
        """Fast partial verification: check valid entries + random off-diagonal sample."""
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
        # Sample 200 random off-diagonal entries
        for _ in range(200):
            u = random.randint(0, nm - 1)
            v = random.randint(0, mp - 1)
            w = random.randint(0, pn - 1)
            i = u // m
            u_j = u % m
            v_j = v // p
            k_v = v % p
            k_w = w // n
            i_w = w % n
            if i == i_w and u_j == v_j and k_v == k_w:
                continue
            val = 0.0
            for r in range(R):
                val += U[u, r] * V[v, r] * W[w, r]
            if abs(val) > 1e-9:
                return False
        return True

    def try_find_rank(R, time_budget=10.0):
        start_time = time.time()
        max_attempts = 3000
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Generate random U and V with controlled sparsity
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(R):
                n_u_nonzero = random.randint(1, 3)
                n_v_nonzero = random.randint(1, 3)
                
                u_indices = random.sample(range(nm), min(n_u_nonzero, nm))
                v_indices = random.sample(range(mp), min(n_v_nonzero, mp))
                
                for idx in u_indices:
                    U[idx, r] = random.choice([-1.0, 1.0])
                for idx in v_indices:
                    V[idx, r] = random.choice([-1.0, 1.0])
            
            # Build coefficient matrix C for W
            C = np.zeros((n*m*p, pn*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i*m*p + j*p + k
                        for r in range(R):
                            u_val = U[i*m+j, r]
                            v_val = V[j*p+k, r]
                            w_idx = k*n + i
                            col = r * pn + w_idx
                            C[e, col] = u_val * v_val
            
            b = np.ones(n*m*p)
            
            try:
                w_flat, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            W = w_flat.reshape(pn, R)
            
            # Round to half-integers
            W_rounded = np.round(W * 2.0) / 2.0
            
            if verify_partial(U, V, W_rounded):
                if verify_full(U, V, W_rounded):
                    return U.tolist(), V.tolist(), W_rounded.tolist()
                
        return None

    # Try ranks from 12 to 17 with adaptive time budget
    for R in range(12, 18):
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 5.0:
            break
        time_budget = min(15.0, remaining - 5.0)
        result = try_find_rank(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Strassen-style rank-17 construction
    # For 2x3 * 3x3, we can use a block decomposition
    # Split A (2x3) into A1 (2x1) and A2 (2x2)
    # Split B (3x3) into B1 (1x3) and B2 (2x3)
    # C = A1*B1 + A2*B2
    # A2*B2 is 2x2 * 2x3, which can be done with rank 7 (Strassen for 2x2*2x2 is 7, but we need 2x2*2x3)
    # Actually, let's just use schoolbook as the safe fallback
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

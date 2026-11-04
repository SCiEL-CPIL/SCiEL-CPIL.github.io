"""Tensor decomposition for 2x3 by 3x9 matrix multiplication."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # 2, 3, 9

    def verify_full(U, V, W):
        """Verify full 2x3 x 3x9 decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def verify_sub(U, V, W, ns, ms, ps):
        """Verify sub-tensor decomposition of size ns x ms x ps."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    u = U_arr[i * ms + j, :]
                    v = V_arr[j * ps + k, :]
                    w = W_arr[k * ns + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def snap(arr):
        return np.round(np.array(arr, dtype=float) * 2) / 2.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_kronecker_from_sub(rank_sub):
        """
        Find rank-rank_sub decomposition for 2x3 x 3x3, then use it
        to build 2x3 x 3x9 with rank 3*rank_sub.
        """
        ns, ms, ps = 2, 3, 3
        
        # The tensor has ns*ms*ps = 18 entries, each should equal 1
        # We need U (ns*ms x rank_sub), V (ms*ps x rank_sub), W (ps*ns x rank_sub)
        # such that for all (i,j,k): sum_r U[i*ms+j, r] * V[j*ps+k, r] * W[k*ns+i, r] = 1
        
        def objective(params):
            """Flatten U, V, W into params and compute sum of squared errors."""
            r = rank_sub
            size_U = ns * ms * r
            size_V = ms * ps * r
            size_W = ps * ns * r
            U_flat = params[:size_U].reshape(ns*ms, r)
            V_flat = params[size_U:size_U+size_V].reshape(ms*ps, r)
            W_flat = params[size_U+size_V:].reshape(ps*ns, r)
            
            errs = []
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        u = U_flat[i*ms+j, :]
                        v = V_flat[j*ps+k, :]
                        w = W_flat[k*ns+i, :]
                        val = np.dot(u * v, w)
                        errs.append(val - 1.0)
            return np.array(errs)
        
        best_result = None
        best_loss = float('inf')
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + rank_sub * 100 + trial)
            
            # Initialize with random small values
            U_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(ns*ms, rank_sub))
            V_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(ms*ps, rank_sub))
            W_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(ps*ns, rank_sub))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(objective, x0, method='lm', max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if result.cost < best_loss:
                    best_loss = result.cost
                    best_result = result
            except Exception:
                continue
        
        if best_result is None or best_result.cost > 1e-6:
            return None
        
        # Extract and snap
        r = rank_sub
        size_U = ns * ms * r
        size_V = ms * ps * r
        size_W = ps * ns * r
        U_sub = snap(best_result.x[:size_U].reshape(ns*ms, r))
        V_sub = snap(best_result.x[size_U:size_U+size_V].reshape(ms*ps, r))
        W_sub = snap(best_result.x[size_U+size_V:].reshape(ps*ns, r))
        
        # Verify the sub-decomposition
        if not verify_sub(U_sub, V_sub, W_sub, ns, ms, ps):
            return None
        
        # Now build the 2x3 x 3x9 decomposition using Kronecker product
        # p = 9 = 3 * 3, so we have 3 blocks
        r_total = 3 * r
        U = [[0] * r_total for _ in range(n*m)]
        V = [[0] * r_total for _ in range(m*p)]
        W = [[0] * r_total for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for blk in range(3):  # 3 blocks of the 9 columns
                    for k_in in range(3):  # 3 columns within each block
                        k = blk * 3 + k_in
                        for c in range(r):
                            col = blk * r + c
                            U[i*m+j][col] = U_sub[i*ms+j, c]
                            V[j*p+k][col] = V_sub[j*ps+k_in, c]
                            W[k*n+i][col] = W_sub[k_in*ns+i, c]
        
        if verify_full(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None

    def try_direct(rank):
        """Direct search for 2x3 x 3x9 with given rank."""
        if time.time() > deadline - 10:
            return None
        
        def objective(params):
            r = rank
            size_U = n * m * r
            size_V = m * p * r
            size_W = p * n * r
            U_flat = params[:size_U].reshape(n*m, r)
            V_flat = params[size_U:size_U+size_V].reshape(m*p, r)
            W_flat = params[size_U+size_V:].reshape(p*n, r)
            
            errs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = U_flat[i*m+j, :]
                        v = V_flat[j*p+k, :]
                        w = W_flat[k*n+i, :]
                        val = np.dot(u * v, w)
                        errs.append(val - 1.0)
            return np.array(errs)
        
        best_result = None
        best_loss = float('inf')
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(2000 + rank * 10 + trial)
            
            U_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, rank))
            V_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, rank))
            W_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, rank))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(objective, x0, method='lm', max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if result.cost < best_loss:
                    best_loss = result.cost
                    best_result = result
            except Exception:
                continue
        
        if best_result is None or best_result.cost > 1e-6:
            return None
        
        r = rank
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        U = snap(best_result.x[:size_U].reshape(n*m, r))
        V = snap(best_result.x[size_U:size_U+size_V].reshape(m*p, r))
        W = snap(best_result.x[size_U+size_V:].reshape(p*n, r))
        
        if verify_full(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try Kronecker product approach with different sub-ranks
    # The sub-problem 2x3 x 3x3 has 18 tensor entries
    # Schoolbook for sub-problem is rank 18
    # Known lower bounds suggest rank ~14-15 is possible
    for r_sub in [14, 15, 16, 17]:
        if time.time() > deadline - 15:
            break
        res = try_kronecker_from_sub(r_sub)
        if res is not None:
            return res
    
    # Try direct search for ranks below 54
    for rank in [45, 48, 50, 52, 53]:
        if time.time() > deadline - 10:
            break
        res = try_direct(rank)
        if res is not None:
            return res

    return schoolbook()
# EVOLVE-BLOCK-END

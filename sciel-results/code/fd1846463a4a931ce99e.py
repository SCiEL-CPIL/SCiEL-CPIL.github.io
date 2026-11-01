import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    target_rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    random.seed(42)
    np.random.seed(42)

    # Precompute target tensor
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0
    target_tensor = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_tensor[u, v, w] = 1.0
    
    # Flatten target
    T_flat = target_tensor.flatten()

    def unpack_params(x, r):
        """
        x: flat array of length r * (nm + mp + pn)
        Returns U, V, W
        """
        size_u = nm * r
        size_v = mp * r
        size_w = pn * r
        
        U = x[:size_u].reshape(nm, r)
        V = x[size_u:size_u+size_v].reshape(mp, r)
        W = x[size_u+size_v:].reshape(pn, r)
        return U, V, W

    def residual_func(x, r):
        U, V, W = unpack_params(x, r)
        # Compute tensor product
        # Res[u, v, w] = sum_t U[u,t] V[v,t] W[w,t]
        Res = np.einsum('ut,vt,wt->uvw', U, V, W)
        return Res.flatten() - T_flat

    def verify(U, V, W):
        r = U.shape[1]
        Res = np.einsum('ut,vt,wt->uvw', U, V, W)
        # Check if Res matches target_tensor
        if not np.allclose(Res, target_tensor, atol=1e-9):
            return False
        return True

    def round_to_half_integer(X):
        return np.round(X * 2) / 2.0

    # Try ranks from schoolbook-1 down to 30
    # Schoolbook is 50. Try 49, 48, ...
    ranks_to_try = list(range(target_rank_schoolbook - 1, 29, -1))
    
    best_result = None
    best_rank = target_rank_schoolbook

    # Precompute schoolbook solution as fallback
    U_school = np.zeros((nm, target_rank_schoolbook))
    V_school = np.zeros((mp, target_rank_schoolbook))
    W_school = np.zeros((pn, target_rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        # Lower ranks are harder, give them more time? Or just try a few.
        # Let's try 50 trials for each rank.
        max_trials = 50
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Initialize parameters
            # Random initialization
            x0 = np.random.randn(r * (nm + mp + pn)) * 0.5
            
            # Optimization
            try:
                # Use least_squares
                # We want to minimize || Res - T ||^2
                # This is a nonlinear least squares problem.
                result = least_squares(residual_func, x0, args=(r,), method='lm', max_nfev=1000)
                
                if result.success or result.cost < 1e-6:
                    U, V, W = unpack_params(result.x, r)
                    
                    # Round to half-integers
                    U_r = round_to_half_integer(U)
                    V_r = round_to_half_integer(V)
                    W_r = round_to_half_integer(W)
                    
                    if verify(U_r, V_r, W_r):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        
                    # If rounding failed, maybe try a different rounding strategy?
                    # Or accept if error is very small? No, must be exact.
                    
            except Exception as e:
                # Optimization might fail
                pass

    # Fallback: Schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END

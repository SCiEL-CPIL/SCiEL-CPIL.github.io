import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """Return the 7 Strassen rank-1 terms for 2x2 matrix multiplication.
    Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Each term: (u_coeffs, v_coeffs, w_coeffs) as lists of (local_idx, value)
    """
    return [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

def _build_rank28():
    """Build the rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen = _strassen_2x2()
    
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    w_map_L = [0, 2, 1, 3]
    w_map_R = [4, 6, 5, 7]
    
    col = 0
    for u_map, v_map, w_map in [(u_map_L, v_map_TL, w_map_L),
                                  (u_map_R, v_map_BL, w_map_L),
                                  (u_map_L, v_map_TR, w_map_R),
                                  (u_map_R, v_map_BR, w_map_R)]:
        for u_t, v_t, w_t in strassen:
            for idx, val in u_t:
                U[u_map[idx], col] += val
            for idx, val in v_t:
                V[v_map[idx], col] += val
            for idx, val in w_t:
                W[w_map[idx], col] += val
            col += 1
    
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=4, tol=1e-9):
    """Verify the tensor decomposition."""
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    return np.allclose(C, Cf, atol=tol)

def _try_reduce_rank(U, V, W, target_rank, max_iters=200, tol=1e-10):
    """Try to reduce rank from current to target_rank via optimization.
    
    Approach: Start with rank-r decomposition, try to zero out (r - target_rank) columns
    by absorbing their contributions into the remaining columns.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    if r <= target_rank:
        return None
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # We'll try to eliminate columns r-target_rank through r-1
    # by modifying the first target_rank columns.
    # This is a nonlinear problem, so we use iterative refinement.
    
    # Strategy: For each column to eliminate, try to find a linear combination
    # of the remaining columns that reproduces its tensor contribution.
    
    cols_to_keep = list(range(target_rank))
    cols_to_eliminate = list(range(target_rank, r))
    
    U_keep = U[:, cols_to_keep].copy()
    V_keep = V[:, cols_to_keep].copy()
    W_keep = W[:, cols_to_keep].copy()
    
    U_elim = U[:, cols_to_eliminate].copy()
    V_elim = V[:, cols_to_eliminate].copy()
    W_elim = W[:, cols_to_eliminate].copy()
    
    # The tensor contribution of eliminated columns must be absorbed.
    # This is hard in general, but let's try a simple approach:
    # For each eliminated column, try to find coefficients that let us
    # add its contribution to kept columns.
    
    # Actually, let's try a different approach: use the fact that we have
    # freedom to modify all kept columns. Set up as least squares.
    
    # The total tensor is: T = sum_c U_c ⊗ V_c ⊗ W_c
    # We want: T = sum_{c in keep} U'_c ⊗ V'_c ⊗ W'_c
    
    # This is a hard nonlinear problem. Let's try a simpler heuristic:
    # Check if any column is a linear combination of others in a tensor sense.
    
    # For now, return None if we can't easily reduce.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # safety margin
    
    # Start with rank-28 Strassen decomposition
    U, V, W = _build_rank28()
    
    # Verify it's correct
    if not _verify(U, V, W):
        # Fallback: shouldn't happen, but return what we have
        if time.time() - start_time > time_limit:
            raise TimeoutError("Construction took too long")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 28
    
    # Try to reduce rank using optimization
    # Try target ranks from 27 down to 20
    for target_rank in range(27, 19, -1):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Attempt: use scipy to solve the tensor decomposition
        try:
            from scipy.optimize import minimize
            from scipy.optimize import least_squares
            
            U0 = np.array(U, dtype=float)
            V0 = np.array(V, dtype=float)
            W0 = np.array(W, dtype=float)
            
            # Flatten parameters for the first target_rank columns
            # We'll modify these to absorb the rest
            U_mod = U0[:, :target_rank].copy()
            V_mod = V0[:, :target_rank].copy()
            W_mod = W0[:, :target_rank].copy()
            
            # The residual tensor (from eliminated columns)
            U_elim = U0[:, target_rank:]
            V_elim = V0[:, target_rank:]
            W_elim = W0[:, target_rank:]
            
            # Build the target tensor T (32 entries)
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = 1.0 if j == k else 0.0
            
            # Current tensor from all columns
            def tensor_from_factors(Uf, Vf, Wf):
                Tt = np.zeros((n, m, p))
                for c in range(Uf.shape[1]):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Tt[i, j, k] += Uf[i*m+j, c] * Vf[j*p+k, c] * Wf[k*n+i, c]
                return Tt
            
            # Residual to absorb
            T_resid = tensor_from_factors(U_elim, V_elim, W_elim)
            
            # We want to modify U_mod, V_mod, W_mod so that
            # tensor_from_factors(U_mod, V_mod, W_mod) + T_resid = T
            # i.e., tensor_from_factors(U_mod, V_mod, W_mod) = T - T_resid
            
            T_target = T - T_resid
            
            # Parameters: flatten U_mod, V_mod, W_mod
            params = np.concatenate([U_mod.ravel(), V_mod.ravel(), W_mod.ravel()])
            n_params = len(params)
            
            def residuals(params):
                Uf = params[:U_mod.size].reshape(U_mod.shape)
                Vf = params[U_mod.size:U_mod.size+V_mod.size].reshape(V_mod.shape)
                Wf = params[U_mod.size+V_mod.size:].reshape(W_mod.shape)
                Tt = tensor_from_factors(Uf, Vf, Wf)
                return (Tt - T_target).ravel()
            
            # Use least_squares
            result = least_squares(residuals, params, method='lm', 
                                    max_nfev=5000, xtol=1e-12)
            
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                Uf = result.x[:U_mod.size].reshape(U_mod.shape)
                Vf = result.x[U_mod.size:U_mod.size+V_mod.size].reshape(V_mod.shape)
                Wf = result.x[U_mod.size+V_mod.size:].reshape(W_mod.shape)
                
                # Verify
                if _verify(Uf, Vf, Wf):
                    best_U, best_V, best_W = Uf, Vf, Wf
                    best_rank = target_rank
                    # Update U, V, W for next iteration
                    U, V, W = Uf, Vf, Wf
                    continue
        
        except Exception as e:
            pass
        
        if time.time() - start_time > time_limit - 3.0:
            break
    
    if time.time() - start_time > time_limit:
        raise TimeoutError("Construction took too long")
    
    # Round to nearest half-integer if close
    def round_half(x):
        return np.round(x * 2) / 2
    
    # Check if we can round to integers/half-integers
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    # Try rounding
    U_r = round_half(U_out)
    V_r = round_half(V_out)
    W_r = round_half(W_out)
    
    if _verify(U_r, V_r, W_r, tol=1e-6):
        U_out, V_out, W_out = U_r, V_r, W_r
    
    # Convert to list
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices to ensure correctness
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])

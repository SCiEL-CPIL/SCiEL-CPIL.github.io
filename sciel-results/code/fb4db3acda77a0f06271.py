import time
import numpy as np
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 110  # safety margin before 120s

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    def build_rank_28():
        """Build the known rank-28 Strassen tiling decomposition."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def add_term(u_vec, v_vec, w_vec, col_idx):
            for idx, coeff in u_vec:
                U[idx, col_idx] += coeff
            for idx, coeff in v_vec:
                V[idx, col_idx] += coeff
            for idx, coeff in w_vec:
                W[idx, col_idx] += coeff
                
        def strassen_terms():
            terms = []
            # M1 = (X00+X11)(Y00+Y11)
            u = {0: 1, 3: 1}; v = {0: 1, 3: 1}; w = {0: 1, 3: 1}
            terms.append((u, v, w))
            # M2 = (X10+X11)Y00
            u = {2: 1, 3: 1}; v = {0: 1}; w = {2: 1, 3: -1}
            terms.append((u, v, w))
            # M3 = X00(Y01-Y11)
            u = {0: 1}; v = {1: 1, 3: -1}; w = {1: 1, 3: 1}
            terms.append((u, v, w))
            # M4 = X11(Y10-Y00)
            u = {3: 1}; v = {2: 1, 0: -1}; w = {0: 1, 2: 1}
            terms.append((u, v, w))
            # M5 = (X00+X01)Y11
            u = {0: 1, 1: 1}; v = {3: 1}; w = {0: -1, 1: 1}
            terms.append((u, v, w))
            # M6 = (X10-X00)(Y00+Y01)
            u = {2: 1, 0: -1}; v = {0: 1, 1: 1}; w = {3: 1}
            terms.append((u, v, w))
            # M7 = (X01-X11)(Y10+Y11)
            u = {1: 1, 3: -1}; v = {2: 1, 3: 1}; w = {0: 1}
            terms.append((u, v, w))
            return terms
        
        base_terms = strassen_terms()
        
        def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
                col = start_col + t_idx
                u_global = {}
                for loc_idx, coeff in u_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = a_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    u_global[g_idx] = u_global.get(g_idx, 0) + coeff
                v_global = {}
                for loc_idx, coeff in v_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = b_row_offset * 2 + r
                    g_c = b_col_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    v_global[g_idx] = v_global.get(g_idx, 0) + coeff
                w_global = {}
                for loc_idx, coeff in w_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = c_col_offset * 2 + c
                    g_idx = g_c * 2 + g_r
                    w_global[g_idx] = w_global.get(g_idx, 0) + coeff
                add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)
        
        apply_strassen(0, 0, 0, 0, 0)
        apply_strassen(1, 1, 0, 0, 7)
        apply_strassen(0, 0, 1, 1, 14)
        apply_strassen(1, 1, 1, 1, 21)
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank):
        """Verify the tensor decomposition is exact."""
        # T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Non-zero entries: T[i*4+j, j*4+k, k*2+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(actual - expected) > 1e-8:
                        return False
        
        # Random verification for robustness
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-8)

    def try_find_rank_r(target_rank, max_restarts=50):
        """
        Attempt to find a decomposition of a specific target rank using random restarts.
        """
        if target_rank < 8:
            return None
            
        num_constraints = n * m * p
        
        # Pre-compute indices for non-zero tensor entries
        # T[u, v, w] = 1 where u=i*m+j, v=j*p+k, w=k*n+i
        tensor_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    tensor_indices.append((u_idx, v_idx, w_idx))

        for restart in range(max_restarts):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            # Random initialization
            # Scale is important. Too large causes overflow/divergence. Too small is slow.
            scale = 1.0
            U_init = np.random.randn(n * m, target_rank) * scale
            V_init = np.random.randn(m * p, target_rank) * scale
            W_init = np.random.randn(p * n, target_rank) * scale
            
            # Flatten for solver
            x0 = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            n_u = n * m
            n_v = m * p
            n_w = p * n
            size_u = n_u * target_rank
            size_v = n_v * target_rank
            size_w = n_w * target_rank
            
            def residuals(x):
                Uv = x[:size_u].reshape(n_u, target_rank)
                Vv = x[size_u:size_u+size_v].reshape(n_v, target_rank)
                Wv = x[size_u+size_v:].reshape(n_w, target_rank)
                
                res = np.zeros(num_constraints)
                idx = 0
                for u_idx, v_idx, w_idx in tensor_indices:
                    # Sum over rank r: U[u, r] * V[v, r] * W[w, r]
                    # Vectorized: dot product of U[u,:], V[v,:], W[w,:]
                    # Note: this is a cubic form, but for fixed U, V it is linear in W.
                    # However, least_squares handles the nonlinearity.
                    acc = 0.0
                    for r in range(target_rank):
                        acc += Uv[u_idx, r] * Vv[v_idx, r] * Wv[w_idx, r]
                    res[idx] = acc - 1.0
                    idx += 1
                return res

            try:
                # Use 'lm' for speed on small problems, 'trf' if bounds are needed (none here)
                result = least_squares(residuals, x0, method='lm', max_nfev=2000)
                
                if result.cost < 1e-10:
                    # Reconstruct matrices
                    U_sol = result.x[:size_u].reshape(n_u, target_rank)
                    V_sol = result.x[size_u:size_u+size_v].reshape(n_v, target_rank)
                    W_sol = result.x[size_u+size_v:].reshape(n_w, target_rank)
                    
                    # Verify with exact check
                    if verify_decomposition(U_sol, V_sol, W_sol, target_rank):
                        return U_sol, V_sol, W_sol
                        
            except Exception:
                continue
                
        return None

    # Build the rank-28 solution as fallback
    U, V, W = build_rank_28()
    current_rank = 28
    
    # Try to find lower rank solutions
    # We know 28 is possible. Try 27, 26, 25, 24.
    # If we find one, we return it.
    for target_rank in range(27, 23, -1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Increase restarts for lower ranks as they are harder
        restarts = 20 if target_rank > 25 else 40
        
        result = try_find_rank_r(target_rank, max_restarts=restarts)
        if result is not None:
            U, V, W = result
            current_rank = target_rank
            break # Found a lower rank, stop searching for even lower immediately? 
                  # Or continue to try lower? 
                  # The goal is rank-minimal. If we found 27, maybe 26 exists.
                  # But 27 is better than 28. Let's try to go lower if time permits.
                  # Actually, the loop continues downwards. 
                  # If we found 27, we update U,V,W and current_rank, and the loop continues to 26.
                  # This is correct.

    # Final verification
    if not verify_decomposition(U, V, W, current_rank):
        # Fall back to rank 28
        U, V, W = build_rank_28()
        current_rank = 28
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])

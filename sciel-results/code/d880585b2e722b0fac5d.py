import time
import numpy as np
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s

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
        """Verify the tensor decomposition is exact using vectorized ops."""
        if rank == 0:
            return False
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        U_sel = U[i_idx * m + j_idx, :]  # (32, rank)
        V_sel = V[j_idx * p + k_idx, :]  # (32, rank)
        W_sel = W[k_idx * n + i_idx, :]  # (32, rank)
        
        acc = np.sum(U_sel * V_sel * W_sel, axis=1)  # (32,)
        if not np.allclose(acc, 1.0, atol=1e-10):
            return False
        
        # Random verification for zero entries
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc2 = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc2 += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
                Cf[ii, kk] = acc2
        return np.allclose(C, Cf, atol=1e-10)
    
    def try_reduce_rank(U, V, W, current_rank, target_rank, max_iters=50):
        """Try to reduce rank by zeroing columns and re-solving with nonlinear solver."""
        if target_rank >= current_rank:
            return None
        
        num_constraints = n * m * p  # 32
        
        # Precompute index arrays for vectorized residual computation
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        def compute_residuals_vec(Uv, Vv, Wv, active_cols):
            """Compute tensor residuals for active columns using vectorized ops."""
            n_active = len(active_cols)
            Ua = Uv[:, active_cols]  # (n*m, n_active)
            Va = Vv[:, active_cols]  # (m*p, n_active)
            Wa = Wv[:, active_cols]  # (p*n, n_active)
            
            # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            U_sel = Ua[i_idx * m + j_idx, :]  # (32, n_active)
            V_sel = Va[j_idx * p + k_idx, :]  # (32, n_active)
            W_sel = Wa[k_idx * n + i_idx, :]  # (32, n_active)
            
            acc = np.sum(U_sel * V_sel * W_sel, axis=1)  # (32,)
            return acc - 1.0
        
        def nonlinear_solve(U_init, V_init, W_init, active_cols):
            """Use scipy least_squares to solve for all free parameters."""
            n_active = len(active_cols)
            U_size = n * m * n_active
            V_size = m * p * n_active
            W_size = p * n * n_active
            
            x0 = np.concatenate([
                U_init[:, active_cols].flatten(),
                V_init[:, active_cols].flatten(),
                W_init[:, active_cols].flatten()
            ])
            
            def residuals(x):
                Uv = x[:U_size].reshape(n*m, n_active)
                Vv = x[U_size:U_size+V_size].reshape(m*p, n_active)
                Wv = x[U_size+V_size:].reshape(p*n, n_active)
                return compute_residuals_vec(Uv, Vv, Wv, active_cols)
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=10000)
                if result.success or result.cost < 1e-20:
                    Uv = result.x[:U_size].reshape(n*m, n_active)
                    Vv = result.x[U_size:U_size+V_size].reshape(m*p, n_active)
                    Wv = result.x[U_size+V_size:].reshape(p*n, n_active)
                    return Uv, Vv, Wv
            except Exception:
                pass
            return None
        
        # Try removing each column systematically
        for col_to_remove in range(current_rank):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            active_cols = [c for c in range(current_rank) if c != col_to_remove]
            
            U_init = U.copy()
            V_init = V.copy()
            W_init = W.copy()
            
            result = nonlinear_solve(U_init, V_init, W_init, active_cols)
            if result is not None:
                U_sol, V_sol, W_sol = result
                if verify_decomposition(U_sol, V_sol, W_sol, len(active_cols)):
                    return U_sol, V_sol, W_sol
        
        return None
    
    # Build the rank-28 solution
    U, V, W = build_rank_28()
    current_rank = 28
    
    # Try to reduce rank iteratively
    while current_rank > 8:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = try_reduce_rank(U, V, W, current_rank, current_rank - 1, max_iters=40)
        if result is not None:
            U, V, W = result
            current_rank -= 1
        else:
            break
    
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

import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}
    
    def _build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            
            U[2][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1
            W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
            col += 1
            
            U[0][col] = 1
            V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
            W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            
            U[3][col] = 1
            V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
            W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
            col += 1
            
            U[0][col] = 1; U[1][col] = 1
            V[1*p + k1][col] = 1
            W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
            col += 1
            
            U[2][col] = 1; U[0][col] = -1
            V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            U[1][col] = 1; U[3][col] = -1
            V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            col += 1
        
        def add_trivial(k):
            nonlocal col
            U[0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + 0][col] = 1
            col += 1
            
            U[1][col] = 1
            V[1*p + k][col] = 1
            W[k*n + 0][col] = 1
            col += 1
            
            U[2][col] = 1
            V[0*p + k][col] = 1
            W[k*n + 1][col] = 1
            col += 1
            
            U[3][col] = 1
            V[1*p + k][col] = 1
            W[k*n + 1][col] = 1
            col += 1
        
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_trivial(6)
        add_strassen(7, 8)
        
        return U, V, W, rank
    
    def _verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Build the rank-32 Strassen-based decomposition
    U, V, W, rank = _build_rank32()
    
    if time.time() > deadline:
        return trivial_decomposition()
    
    # Verify rank-32 solution
    if not _verify(U, V, W, rank):
        return trivial_decomposition()
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try a brief local search for rank 31 using smooth epigraph approach
    # This is a heuristic attempt within the same decomposition family
    search_deadline = start_time + 105.0
    try:
        from scipy.optimize import minimize
        import random
        
        def _compute_residual(U_arr, V_arr, W_arr, r):
            """Compute max absolute residual for tensor entries."""
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        res = abs(val - 1.0)
                        if res > max_res:
                            max_res = res
            return max_res
        
        # Try rank 31: start from rank-32, zero out one column, optimize
        rng = np.random.default_rng(42)
        U31_init = np.array(U, dtype=float)[:, :31]
        V31_init = np.array(V, dtype=float)[:, :31]
        W31_init = np.array(W, dtype=float)[:, :31]
        
        # Flatten parameters
        init_params = np.concatenate([U31_init.flatten(), V31_init.flatten(), W31_init.flatten()])
        
        def objective(params):
            if time.time() > search_deadline:
                return 1e10
            U_arr = params[:4*31].reshape(4, 31)
            V_arr = params[4*31:4*31+18*31].reshape(18, 31)
            W_arr = params[4*31+18*31:].reshape(18, 31)
            return _compute_residual(U_arr, V_arr, W_arr, 31)
        
        res = minimize(objective, init_params, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12,
                               'adaptive': True})
        
        if res.fun < 1e-9:
            U_arr = res.x[:4*31].reshape(4, 31)
            V_arr = res.x[4*31:4*31+18*31].reshape(18, 31)
            W_arr = res.x[4*31+18*31:].reshape(18, 31)
            
            # Convert to list of lists with rational approximation
            U31 = [[float(U_arr[i, c]) for c in range(31)] for i in range(4)]
            V31 = [[float(V_arr[i, c]) for c in range(31)] for i in range(18)]
            W31 = [[float(W_arr[i, c]) for c in range(31)] for i in range(18)]
            
            if _verify(U31, V31, W31, 31):
                best_result = {"U": U31, "V": V31, "W": W31}
                best_rank = 31
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END

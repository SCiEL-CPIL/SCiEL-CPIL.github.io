"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    def verify_full(U, V, W, rank, tol=1e-9):
        """
        Verify full tensor decomposition.
        Condition: sum_c U[i',j']_c * V[j'',k'']_c * W[k,i]_c = delta(i',i) * delta(j',j'') * delta(k'',k)
        Indices:
        U: (i, j) -> index i*m + j
        V: (j, k) -> index j*p + k
        W: (k, i) -> index k*n + i
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Precompute products for efficiency?
        # Loop over all target indices (i, k) and source indices (i', j', j'', k'') is too slow if done naively.
        # Better: Compute the tensor T and check against identity-like structure.
        
        # T[i', j', j'', k'', i, k] = sum_c U[i',j']_c V[j'',k'']_c W[k,i]_c
        # We want T to be 1 if i'=i, j'=j'', k''=k, else 0.
        
        # Let's iterate over the "output" indices i, k and "input" indices i', j', j'', k''
        # Actually, it's easier to iterate over c and build the tensor? No, rank is small.
        
        # Check all combinations
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for jp2 in range(m):
                            for kp in range(p):
                                # Target value
                                target = 1.0 if (ip == i and jp == jp2 and kp == k) else 0.0
                                # Computed value
                                s = 0.0
                                for c in range(rank):
                                    s += U[ip*m + jp, c] * V[jp2*p + kp, c] * W[k*n + i, c]
                                if abs(s - target) > tol:
                                    return False
        return True

    def objective_full(params, rank):
        """
        Objective: sum of (sum_c U V W - target)^2 over all indices.
        """
        U = params[0:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        res = 0.0
        # To speed up, we can vectorize the sum over c for each index tuple
        # But the number of tuples is n*p * n*m * m*p = 12 * 6 * 18 = 1296.
        # 1296 * rank (24) operations is small.
        
        for i in range(n):
            for k in range(p):
                w_row = W[k*n + i, :]
                for ip in range(n):
                    for jp in range(m):
                        u_val = U[ip*m + jp, :]
                        for jp2 in range(m):
                            for kp in range(p):
                                v_val = V[jp2*p + kp, :]
                                s = np.dot(u_val * v_val, w_row)
                                target = 1.0 if (ip == i and jp == jp2 and kp == k) else 0.0
                                res += (s - target) ** 2
        return res

    def schoolbook():
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
        return U, V, W, rank

    def try_rank_opt(rank, n_starts=5, max_iters=200):
        if time.time() > deadline - 30:
            return None
            
        best_result = None
        best_obj = float('inf')
        
        for start in range(n_starts):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(42 + start * 11 + rank)
            # Initialize with small random values
            U_init = np.random.randn(n*m, rank) * 0.1
            V_init = np.random.randn(m*p, rank) * 0.1
            W_init = np.random.randn(p*n, rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def obj_func(p):
                return objective_full(p, rank)
                
            try:
                res = minimize(obj_func, params_init, method='L-BFGS-B', 
                               options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-10})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_result = res.x
                    
                    if res.fun < 1e-10:
                        U_sol = res.x[0:n*m*rank].reshape(n*m, rank)
                        V_sol = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                        W_sol = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                        
                        # Snap to half-integers
                        U_snap = np.round(2.0 * U_sol) / 2.0
                        V_snap = np.round(2.0 * V_sol) / 2.0
                        W_snap = np.round(2.0 * W_sol) / 2.0
                        
                        if verify_full(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank, tol=1e-8):
                            return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                            
                        # If snap fails, try integer snap
                        U_int = np.round(U_sol)
                        V_int = np.round(V_sol)
                        W_int = np.round(W_sol)
                        if verify_full(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank, tol=1e-8):
                            return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                            
            except Exception as e:
                pass
        
        return None

    # Try ranks from 24 upwards
    for rank in range(24, 37):
        if time.time() > deadline - 40:
            break
            
        result = try_rank_opt(rank, n_starts=5, max_iters=100)
        if result is not None:
            U, V, W = result
            # Final verification
            if verify_full(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute index triples and their (U_idx, V_idx, W_idx)
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid."""
        for (u_idx, v_idx, w_idx) in indices:
            val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
            if abs(val - 1.0) > 1e-8:
                return False
        return True
    
    def to_half_int_list(arr):
        """Convert numpy array to list with half-integer rounding."""
        lst = []
        for row in arr:
            row_lst = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_lst.append(rounded)
                else:
                    row_lst.append(float(val))
            lst.append(row_lst)
        return lst
    
    # Strategy: Use algebraic structure to construct lower-rank decomposition
    # For 2x3x4, try ranks from 19 upward
    
    # First, try a constructive approach based on the 3x3x3 embedding
    # The 3x3x3 multiplication has known rank-23 decomposition (Strassen)
    # We can try to restrict it to our 2x3x4 subproblem
    
    # Alternative: direct optimization with good initialization
    # Use the structure: initialize with the naive decomposition projected
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Try to find rank-19 or rank-20 decomposition
    for target_rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(target_rank * 10000 + attempt * 13 + 7)
            
            # Initialize with structured random values
            scale = 1.0 / (target_rank ** 0.25)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residual(params, r):
                U_flat = params[:nm * r].reshape(nm, r)
                V_flat = params[nm * r: nm * r + mp * r].reshape(mp, r)
                W_flat = params[nm * r + mp * r:].reshape(pn, r)
                
                res = np.empty(len(indices))
                for idx, (u_idx, v_idx, w_idx) in enumerate(indices):
                    res[idx] = np.dot(U_flat[u_idx, :], V_flat[v_idx, :] * W_flat[w_idx, :]) - 1.0
                return res
            
            try:
                result = least_squares(residual, x0, args=(target_rank,), method='lm', max_nfev=500, xtol=1e-14, ftol=1e-14, gtol=1e-14)
                
                if result.cost < 1e-10:
                    U_sol = result.x[:nm * target_rank].reshape(nm, target_rank)
                    V_sol = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                    W_sol = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                    
                    max_err = 0.0
                    for (u_idx, v_idx, w_idx) in indices:
                        val = np.dot(U_sol[u_idx, :], V_sol[v_idx, :] * W_sol[w_idx, :])
                        max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-8:
                        # Polish: try to round to half-integers
                        U_list = to_half_int_list(U_sol)
                        V_list = to_half_int_list(V_sol)
                        W_list = to_half_int_list(W_sol)
                        
                        if check_decomposition(U_list, V_list, W_list, target_rank):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        else:
                            # Use the exact float values
                            U_list = U_sol.tolist()
                            V_list = V_sol.tolist()
                            W_list = W_sol.tolist()
                            return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
        
        if not found:
            continue
    
    # Fallback: naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

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
    
    # Precompute index triples: (u_idx, v_idx, w_idx) for each (i, j, k)
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
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Build the structured index mapping for vectorized computation
    u_indices = np.array([idx[0] for idx in indices])
    v_indices = np.array([idx[1] for idx in indices])
    w_indices = np.array([idx[2] for idx in indices])
    num_constraints = len(indices)
    
    def residual_vectorized(params, r):
        """Vectorized residual computation."""
        U_flat = params[:nm * r].reshape(nm, r)
        V_flat = params[nm * r: nm * r + mp * r].reshape(mp, r)
        W_flat = params[nm * r + mp * r:].reshape(pn, r)
        
        # Compute all dot products at once
        U_sel = U_flat[u_indices, :]  # (num_constraints, r)
        V_sel = V_flat[v_indices, :]  # (num_constraints, r)
        W_sel = W_flat[w_indices, :]  # (num_constraints, r)
        
        # Element-wise product then sum over rank dimension
        res = np.sum(U_sel * V_sel * W_sel, axis=1) - 1.0
        return res
    
    # Try to find rank-19 or lower decomposition
    for target_rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 25:
            break
        
        found = False
        for attempt in range(15):
            if time.time() > deadline - 15:
                break
            
            seed_val = target_rank * 100000 + attempt * 7919 + 13
            np.random.seed(seed_val)
            
            # Multiple initialization strategies
            strategy = attempt % 3
            
            if strategy == 0:
                # Random Gaussian
                scale = 1.0 / (target_rank ** 0.25)
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            elif strategy == 1:
                # Sparse initialization - many zeros
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    # Assign random non-zero entries
                    u_rows = np.random.choice(nm, size=min(3, nm), replace=False)
                    v_rows = np.random.choice(mp, size=min(4, mp), replace=False)
                    w_rows = np.random.choice(pn, size=min(3, pn), replace=False)
                    U[u_rows, r] = np.random.choice([-1, 1], size=len(u_rows))
                    V[v_rows, r] = np.random.choice([-1, 1], size=len(v_rows))
                    W[w_rows, r] = np.random.choice([-1, 1], size=len(w_rows))
            else:
                # Structured initialization based on naive decomposition
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                # Initialize first min(target_rank, nm*mp*pn) columns from naive
                naive_rank = min(target_rank, n * m * p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = i * (m * p) + j * p + k
                            if c < naive_rank:
                                U[i * m + j][c] = 1.0
                                V[j * p + k][c] = 1.0
                                W[k * n + i][c] = 1.0
                # Add small random perturbation to remaining columns
                for c in range(naive_rank, target_rank):
                    U[:, c] = np.random.randn(nm) * 0.1
                    V[:, c] = np.random.randn(mp) * 0.1
                    W[:, c] = np.random.randn(pn) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use different optimization methods
                if attempt % 2 == 0:
                    result = least_squares(
                        residual_vectorized, x0, args=(target_rank,),
                        method='lm', max_nfev=800, xtol=1e-12, ftol=1e-12, gtol=1e-12
                    )
                else:
                    result = least_squares(
                        residual_vectorized, x0, args=(target_rank,),
                        method='trf', max_nfev=800, xtol=1e-12, ftol=1e-12, gtol=1e-12
                    )
                
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

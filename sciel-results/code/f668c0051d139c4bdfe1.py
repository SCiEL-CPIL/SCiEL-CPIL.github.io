"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def get_error(params, R):
        """
        Compute residuals for a given set of parameters.
        params: flat array of length R * (n*m + m*p + p*n)
        """
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        
        residuals = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_idx = k * n + i
                    u_idx = i * m + j
                    v_idx = j * p + k
                    
                    val = np.dot(W[:, w_idx], U[:, u_idx] * V[:, v_idx])
                    target = 1.0 if j == k else 0.0
                    residuals[idx] = val - target
                    idx += 1
        return residuals

    def snap_and_check(params, R):
        """
        Snap params to nearest half-integer and check validity.
        Returns (U, V, W) as lists if valid, else None.
        """
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Reconstruct flat params for verification
        params_snapped = np.concatenate([U_snapped.flatten(), V_snapped.flatten(), W_snapped.flatten()])
        
        errs = get_error(params_snapped, R)
        if np.max(np.abs(errs)) < 1e-9:
            # Convert to lists (transposed to match (dim, rank) shape)
            U_list = U_snapped.T.tolist()
            V_list = V_snapped.T.tolist()
            W_list = W_snapped.T.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    def get_fallback():
        """
        Construct the known rank 44 solution.
        """
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        # Part 1: 16 terms for A[:, 0] * B[0, :]
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        # Part 2: 4 blocks of 7 terms for the remaining 2x2x2 like structures
        # This part is copied from the original code's logic
        for block_idx in range(4):
            k_start = block_idx * 2
            steps = [
                {'U': {0*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1}, 'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}},
                {'U': {0*m + 1: 1}, 'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1}, 'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 2: 1}, 'V': {2*p + k_start: 1, 1*p + k_start: -1}, 'W': {k_start*n + 0: 1, k_start*n + 1: 1}},
                {'U': {0*m + 1: 1, 0*m + 2: 1}, 'V': {2*p + (k_start+1): 1}, 'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}},
                {'U': {1*m + 1: 1, 0*m + 1: -1}, 'V': {1*p + k_start: 1, 1*p + (k_start+1): 1}, 'W': {(k_start+1)*n + 1: 1}},
                {'U': {0*m + 2: 1, 1*m + 2: -1}, 'V': {2*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1}}
            ]
            for step in steps:
                for u_idx, val in step['U'].items():
                    U[u_idx][col] += val
                for v_idx, val in step['V'].items():
                    V[v_idx][col] += val
                for w_idx, val in step['W'].items():
                    W[w_idx][col] += val
                col += 1
        return {"U": U, "V": V, "W": W}

    def lists_to_flat(solution, R):
        """Convert dict of lists to flat numpy array."""
        U = np.array(solution['U']) # shape (n*m, R)
        V = np.array(solution['V']) # shape (m*p, R)
        W = np.array(solution['W']) # shape (p*n, R)
        
        # Transpose to (R, dim) for internal use
        U_flat = U.T.flatten()
        V_flat = V.T.flatten()
        W_flat = W.T.flatten()
        
        return np.concatenate([U_flat, V_flat, W_flat])

    def try_reduce_rank(current_solution, R_target):
        """
        Try to reduce rank of current_solution to R_target by removing one term
        and re-optimizing the rest.
        """
        R_current = len(current_solution['U'][0])
        if R_target >= R_current:
            return None
        
        flat_current = lists_to_flat(current_solution, R_current)
        
        # We need to try removing each of the R_current terms
        for k in range(R_current):
            if time.time() - start_time > timeout:
                return None
                
            # Construct initial guess for R_target terms
            # We take all terms except k
            # We need to reshape flat_current to (R_current, total_dim)
            size_U = n * m
            size_V = m * p
            size_W = p * n
            total_dim = size_U + size_V + size_W
            
            matrix_current = flat_current.reshape(R_current, total_dim)
            
            # Remove row k
            if k == 0:
                matrix_reduced = matrix_current[1:]
            elif k == R_current - 1:
                matrix_reduced = matrix_current[:-1]
            else:
                matrix_reduced = np.concatenate([matrix_current[:k], matrix_current[k+1:]])
            
            x0 = matrix_reduced.flatten()
            
            try:
                res = least_squares(
                    get_error, 
                    x0, 
                    args=(R_target,), 
                    method='lm', 
                    max_nfev=10000,
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15
                )
            except Exception:
                continue
                
            if res.success:
                snapped = snap_and_check(res.x, R_target)
                if snapped is not None:
                    return snapped
        
        return None

    # Start with rank 44
    best_solution = get_fallback()
    current_rank = 44
    
    # Try to reduce rank
    for R_target in range(43, 39, -1): # Try 43, 42, 41, 40
        if time.time() - start_time > timeout:
            break
            
        reduced = try_reduce_rank(best_solution, R_target)
        if reduced is not None:
            best_solution = reduced
            current_rank = R_target
            # Continue trying to reduce further
        else:
            # If we can't reduce to R_target, maybe we can't reduce further?
            # It's possible we need to remove a different set of terms, but 
            # removing one term at a time is a good heuristic.
            # We break because if we can't go from R+1 to R, it's unlikely 
            # we can go from R+1 to R-1 directly without more complex moves.
            break
            
    return best_solution
# EVOLVE-BLOCK-END

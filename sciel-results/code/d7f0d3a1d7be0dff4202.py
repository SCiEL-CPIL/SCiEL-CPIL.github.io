"""Improved program: attempts a lower-rank exact decomposition for 3x3*3x6."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def check_decomp(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=1e-10):
                        return False
        return True

    def make_residuals(rank):
        """Create a residual function for a given rank."""
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        def residuals(params):
            U = params[0:size_U].reshape(n * m, rank)
            V = params[size_U:size_U + size_V].reshape(m * p, rank)
            W = params[size_U + size_V:].reshape(p * n, rank)
            
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        res[idx] = val - 1.0
                        idx += 1
            return res
        
        return residuals

    def find_decomp_for_rank(rank, max_attempts, time_per_attempt):
        """Try to find a half-integer decomposition for the given rank."""
        residuals_fn = make_residuals(rank)
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Try different initialization strategies
            if attempt % 3 == 0:
                # Small random init
                params = np.random.randn(total_size) * 0.5
            elif attempt % 3 == 1:
                # Structured init: start with identity-like structure
                params = np.zeros(total_size)
                # Initialize with a pattern that might help
                for c in range(rank):
                    if c < n * m:
                        params[c] = 1.0  # U
                    elif c < n * m + m * p:
                        params[size_U + (c - n * m)] = 1.0  # V
                    else:
                        params[size_U + size_V + (c - n * m - m * p)] = 1.0  # W
                # Add small random perturbation
                params += np.random.randn(total_size) * 0.01
            else:
                # Medium random init
                params = np.random.randn(total_size) * 1.0
            
            # Use least_squares with tight tolerances
            try:
                result = least_squares(
                    residuals_fn, params,
                    method='lm',
                    xtol=1e-14, ftol=1e-14, gtol=1e-14,
                    max_nfev=5000
                )
            except Exception:
                continue
            
            if result.cost < 1e-12:
                U = result.x[0:size_U].reshape(n * m, rank)
                V = result.x[size_U:size_U + size_V].reshape(m * p, rank)
                W = result.x[size_U + size_V:].reshape(p * n, rank)
                
                # Try rounding to half-integers
                for round_factor in [2, 4, 8, 16]:
                    U_r = np.round(U * round_factor) / round_factor
                    V_r = np.round(V * round_factor) / round_factor
                    W_r = np.round(W * round_factor) / round_factor
                    
                    if check_decomp(U_r, V_r, W_r):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        
        return None

    # Try ranks from low to high, with decreasing time budget per rank
    # Start with the theoretical lower bound
    for rank in [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]:
        if time.time() > deadline - 5:
            break
        
        # Allocate more attempts for lower ranks
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
        
        # Give lower ranks more attempts
        if rank <= 46:
            max_attempts = 20
            time_per_attempt = remaining_time / max_attempts
        elif rank <= 49:
            max_attempts = 10
            time_per_attempt = remaining_time / max_attempts
        else:
            max_attempts = 5
            time_per_attempt = remaining_time / max_attempts
        
        decomp = find_decomp_for_rank(rank, max_attempts, time_per_attempt)
        if decomp:
            return decomp
    
    # Fallback to schoolbook 54
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
    return {"U": U, "V": V, "W": W}

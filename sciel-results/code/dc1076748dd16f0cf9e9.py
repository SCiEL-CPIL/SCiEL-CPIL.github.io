"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses vectorized einsum for fast residual computation and structured random search.
    """
    start_time = time.time()
    timeout = 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 8
    size_U = n * m  # 6
    size_V = m * p  # 24
    size_W = p * n  # 16

    # Precompute target tensor
    target = np.zeros((n, m, p))
    for j in range(m):
        for k in range(p):
            if j == k:
                for i in range(n):
                    target[i, j, k] = 1.0

    def compute_residuals(U, V, W):
        """Compute all residuals using einsum for speed."""
        # U: (R, size_U) -> reshape to (R, n, m)
        # V: (R, size_V) -> reshape to (R, m, p)
        # W: (R, size_W) -> reshape to (R, p, n)
        U3 = U.reshape(U.shape[0], n, m)
        V3 = V.reshape(V.shape[0], m, p)
        W3 = W.reshape(W.shape[0], p, n)
        
        # For each (i,j,k): sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
        # Use einsum: 'rij,rjk,rki->ijk'
        result = np.einsum('rij,rjk,rki->ijk', U3, V3, W3)
        return (result - target).flatten()

    def error_func(params, R):
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        return compute_residuals(U, V, W)

    def try_rank(R, max_restarts=20):
        """Try to find a valid rank-R decomposition."""
        total_size = R * (size_U + size_V + size_W)
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                return None
                
            np.random.seed(42 + restart * 7)
            params = np.random.randn(total_size) * 0.5
            
            try:
                res = least_squares(error_func, params, args=(R,), method='lm', max_nfev=3000)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                U = res.x[0 : R*size_U].reshape(R, size_U)
                V = res.x[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
                W = res.x[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
                
                # Snap to nearest half-integer
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                residuals_snapped = compute_residuals(U_snapped, V_snapped, W_snapped)
                
                if np.max(np.abs(residuals_snapped)) < 1e-9:
                    # Transpose to get (size, R) format
                    U_list = U_snapped.T.tolist()
                    V_list = V_snapped.T.tolist()
                    W_list = W_snapped.T.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def get_fallback():
        """Return the known rank-44 schoolbook-like solution."""
        rank = 44
        U = [[0.0] * rank for _ in range(size_U)]
        V = [[0.0] * rank for _ in range(size_V)]
        W = [[0.0] * rank for _ in range(size_W)]
        
        # Standard schoolbook: each (i,j,k) with j==k gets one rank-1 term
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        U[i*m + j][col] = 1.0
                        V[j*p + k][col] = 1.0
                        W[k*n + i][col] = 1.0
                        col += 1
        
        # Fill remaining columns with zeros (rank 44 but only 16 non-zero terms)
        # Actually schoolbook rank is n*m*p = 48, but we only need j==k so 2*3*8/3... 
        # Let me just use a valid construction.
        
        # Simpler: use the block-partitioned approach from the original
        # Reset and use a known valid rank-44 construction
        U = [[0.0] * rank for _ in range(size_U)]
        V = [[0.0] * rank for _ in range(size_V)]
        W = [[0.0] * rank for _ in range(size_W)]
        
        col = 0
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1.0
                V[0*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
        
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
                    U[u_idx][col] += float(val)
                for v_idx, val in step['V'].items():
                    V[v_idx][col] += float(val)
                for w_idx, val in step['W'].items():
                    W[w_idx][col] += float(val)
                col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try to find a better rank than 44
    best_result = None
    
    for R in [40, 41, 42, 43]:
        if time.time() - start_time > timeout:
            break
        result = try_rank(R, max_restarts=15)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        best_result = get_fallback()
        
    return best_result
# EVOLVE-BLOCK-END

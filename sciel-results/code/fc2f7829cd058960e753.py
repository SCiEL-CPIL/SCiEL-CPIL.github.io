import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # Target rank 13
    target_rank = 13
    
    # Build the target tensor T of shape (u_dim, v_dim, w_dim)
    # T[i, j, k] corresponds to A[i//m, i%m] * B[i%m, j] contributing to C[j//m, j%m]
    # Wait, indices:
    # A is n x m. Index u = i*m + j. (i in 0..n-1, j in 0..m-1)
    # B is m x p. Index v = j*p + k. (j in 0..m-1, k in 0..p-1)
    # C is p x n. Index w = k*n + i. (k in 0..p-1, i in 0..n-1)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    
    T = np.zeros((u_dim, v_dim, w_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Try to find a rank 13 decomposition
    # We use a simple iterative solver or a random search with local optimization.
    # Given the small size, we can try to solve for U, V, W.
    
    # Initialize with rank 14 solution (Strassen x 2) and then try to reduce?
    # Or just random init and least squares?
    
    # Let's try a structured initialization based on Strassen.
    # Strassen 7 terms for 2x2x2.
    # We will try to fit 13 terms.
    
    # If time permits, use scipy.optimize
    try:
        from scipy.optimize import least_squares
        import random
        
        start_time = time.time()
        time_limit = 100  # seconds
        
        # Random initialization
        rank = target_rank
        U = np.random.randn(u_dim, rank)
        V = np.random.randn(v_dim, rank)
        W = np.random.randn(w_dim, rank)
        
        # Normalize to prevent scaling issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            if norm_u > 1e-9:
                U[:, r] /= norm_u
                V[:, r] *= norm_u
            norm_v = np.linalg.norm(V[:, r])
            if norm_v > 1e-9:
                V[:, r] /= norm_v
                W[:, r] *= norm_v
                
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
        def unflatten_params(x):
            U_size = u_dim * rank
            V_size = v_dim * rank
            U = x[:U_size].reshape(u_dim, rank)
            V = x[U_size:U_size+V_size].reshape(v_dim, rank)
            W = x[U_size+V_size:].reshape(w_dim, rank)
            return U, V, W
            
        def residuals(x):
            U, V, W = unflatten_params(x)
            # Compute T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # Residual is T - T_approx
            # To be efficient, we can compute the tensor product
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            return (T - T_approx).flatten()
            
        # Optimize
        x0 = flatten_params(U, V, W)
        result = None
        best_cost = np.inf
        
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=10000)
                cost = res.cost
                if cost < best_cost:
                    best_cost = cost
                    result = res
                    if cost < 1e-8:
                        break
            except Exception:
                pass
                
        if result is not None and best_cost < 1e-6:
            U, V, W = unflatten_params(result.x)
            # Convert to lists of integers/half-integers if possible
            # The problem asks for integer/half-integer entries.
            # Numerical solutions might be floating point.
            # We need to round them.
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness
            T_approx = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
            if np.allclose(T, T_approx, atol=1e-9):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
                
    except ImportError:
        pass
        
    # Fallback to Rank 14
    return _build_rank14()

def _build_rank14():
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for t in range(7):
        u_dict, v_dict, w_dict = strassen_terms[t]
        col = t
        for (i, j), coeff in u_dict.items():
            U[i*m+j][col] += coeff
        for (j, kl), coeff in v_dict.items():
            V[j*p+kl][col] += coeff
        for (kl, i), coeff in w_dict.items():
            W[kl*n+i][col] += coeff
            
    for t in range(7):
        u_dict, v_dict, w_dict = strassen_terms[t]
        col = 7 + t
        for (i, j), coeff in u_dict.items():
            U[i*m+j][col] += coeff
        for (j, kl), coeff in v_dict.items():
            V[j*p+(kl+2)][col] += coeff
        for (kl, i), coeff in w_dict.items():
            W[(kl+2)*n+i][col] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

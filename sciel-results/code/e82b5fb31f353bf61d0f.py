"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U.reshape(n, m, -1)
    V_t = V.reshape(m, p, -1)
    W_t = W.reshape(p, n, -1)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Target is all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _objective_vectorized(params, n, m, p, rank):
    """Calculate the sum of squared errors for the tensor decomposition."""
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:size_U + size_V + size_W]
    
    U = U_flat.reshape((n, m, rank))
    V = V_flat.reshape((m, p, rank))
    W = W_flat.reshape((p, n, rank))
    
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    
    error = np.sum((T - 1.0) ** 2)
    return error

def _try_rank(rank, n, m, p, max_time=10.0, seed=42):
    """Try to find a rank-rank decomposition using scipy.optimize.minimize."""
    rng = np.random.RandomState(seed)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    # Structured Initialization: 
    # Start with a "redundant" schoolbook-like structure.
    # Map each (i,j,k) to a column. If rank < n*m*p, we have collisions.
    # We can initialize by taking the first 'rank' columns of a random permutation of schoolbook columns.
    # Or simply random initialization with small noise is often better for L-BFGS if we start from a known good point.
    # Let's try initializing with the schoolbook decomposition truncated/padded if possible, or just random.
    # Given the high non-convexity, random restarts are key.
    
    params = rng.randn(total_size) * 0.1
    
    # Add a small bias to encourage non-zero values
    # Actually, let's try a specific initialization: 
    # If rank >= n*m*p, we can use exact schoolbook.
    # If rank < n*m*p, we need to merge.
    # Let's stick to random but use multiple seeds.
    
    start_time = time.time()
    
    # Use L-BFGS-B
    try:
        result = minimize(_objective_vectorized, params, args=(n, m, p, rank), method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-8, 'maxfun': 1000, 'maxls': 50})
    except Exception:
        return None
    
    if time.time() - start_time > max_time:
        return None
        
    # Check if the result is good enough
    final_error = result.fun
    if final_error < 1e-10:
        # Extract U, V, W
        U_flat = result.x[:size_U]
        V_flat = result.x[size_U:size_U + size_V]
        W_flat = result.x[size_U + size_V:size_U + size_V + size_W]
        
        U = U_flat.reshape((n*m, rank)).tolist()
        V = V_flat.reshape((m*p, rank)).tolist()
        W = W_flat.reshape((p*n, rank)).tolist()
        
        # Final verification
        if _verify(U, V, W, n, m, p):
            return (U, V, W)
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
                
    # Known lower bounds/estimates for 2x3x4 is around 16.
    # We try ranks from 16 up to schoolbook_rank - 1.
    # If we find one, we stop.
    
    # To save time, we only try a limited number of seeds per rank.
    # The vectorized objective is much faster, allowing more iterations.
    
    for current_rank in range(16, schoolbook_rank):
        if time.time() > _DEADLINE - 5:
            break
            
        found = False
        # Try a few seeds
        for seed in range(20):
            if time.time() > _DEADLINE - 5:
                break
                
            result = _try_rank(current_rank, n, m, p, max_time=5.0, seed=seed)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = current_rank
                found = True
                break
        
        if found:
            # If we found a solution, we can try to go lower? 
            # The loop goes UP from 16. If we find 16, we stop.
            # If we want to be sure 16 is minimal, we'd need to prove 15 is impossible.
            # The task asks for "rank-minimal". 
            # However, finding a valid 16 is a big improvement over 24.
            # If we find 16, we can return it.
            # If we don't find 16, we try 17, etc.
            # To be safe and fast, we return the first one found.
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

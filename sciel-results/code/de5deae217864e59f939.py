"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Actually the convention is: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the full tensor T of shape (n, m, p) where T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Greedy algorithm: iteratively pick the best rank-1 tensor to subtract
    # A rank-1 tensor is defined by vectors u (n*m), v (m*p), w (p*n)
    # such that T_residual[i,j,k] approx = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # We'll try to find integer/half-integer solutions
    # Start with a reasonable upper bound on rank
    # For 3x3*3x8, known bounds suggest rank might be significantly below 72
    
    # Greedy approach: at each step, find the best rank-1 update
    # We parameterize by trying small integer values
    
    max_rank_to_try = min(60, rank_school)  # Try to find solution with rank <= 60
    
    for target_rank in range(24, max_rank_to_try + 1):
        if time.time() > deadline - 20:
            break
            
        # Try greedy construction with target_rank terms
        success = _greedy_construct(T, n, m, p, target_rank, deadline)
        if success is not None:
            U, V, W = success
            if target_rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                if best_rank <= 40:  # Good enough, stop early
                    break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _greedy_construct(T, n, m, p, target_rank, deadline):
    """
    Greedy algorithm to find a rank-`target_rank` decomposition.
    Returns (U, V, W) arrays or None if failed.
    """
    T_residual = T.copy().astype(float)
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Precompute all possible rank-1 tensor patterns
    # A rank-1 tensor is u[i*m+j] * v[j*p+k] * w[k*n+i]
    # We restrict to small integer values: {-1, 0, 1} for simplicity first
    
    # Actually, let's use a more systematic approach
    # Try all combinations of small integer vectors
    
    values = [-1, 0, 1]
    
    for r in range(target_rank):
        if time.time() > deadline - 10:
            return None
        
        best_score = -1e18
        best_u, best_v, best_w = None, None, None
        
        # We need to find u, v, w that maximize correlation with T_residual
        # This is equivalent to finding the best rank-1 approximation
        
        # For efficiency, fix u and w, optimize v, etc.
        # But that's complex. Let's use a simpler heuristic:
        
        # Try random initializations and local search
        np.random.seed(42 + r)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization with small integers
            u = np.random.choice(values, size=n*m)
            v = np.random.choice(values, size=m*p)
            w = np.random.choice(values, size=p*n)
            
            # Compute the rank-1 tensor
            tensor = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            score = np.sum(T_residual * tensor)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None or best_score <= 1e-10:
            return None
        
        # Subtract the best rank-1 tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_residual[i,j,k] -= best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]
        
        U[:, r] = best_u
        V[:, r] = best_v
        W[:, r] = best_w
        
        # Check if residual is close to zero
        if np.allclose(T_residual, 0, atol=1e-10):
            # Success! Trim unused columns
            for r2 in range(r+1, target_rank):
                U[:, r2] = 0
                V[:, r2] = 0
                W[:, r2] = 0
            return U[:, :r+1], V[:, :r+1], W[:, :r+1]
    
    # Check final residual
    if np.allclose(T_residual, 0, atol=1e-10):
        return U, V, W
    
    return None

# EVOLVE-BLOCK-END

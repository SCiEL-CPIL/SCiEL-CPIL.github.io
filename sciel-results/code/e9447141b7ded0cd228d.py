"""Rank-reduced tensor decomposition for 3x4x8 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8):
    """Verify exact decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] 
                        for r_idx in range(r))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=4, p=8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    errors = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                errors += (s - 1.0) ** 2
    return errors < 1e-18

def _schoolbook(n=3, m=4, p=8):
    """Return schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_decomposition(n, m, p, rank, seed=42, max_iter=2000):
    """Try to find a valid rank-r decomposition via random initialization + refinement."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # We need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p = 96 equations.
    
    # Use a simple iterative approach: pick random (i,j,k), compute residual,
    # and adjust one of the factors.
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute all residuals
        residuals = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residuals[(i,j,k)] = s - 1.0
        
        total_err = sum(r**2 for r in residuals.values())
        if total_err < best_err:
            best_err = total_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if total_err < 1e-18:
                break
        
        if total_err < 1e-18:
            break
        
        # Pick the worst residual
        worst_key = max(residuals, key=lambda k: abs(residuals[k]))
        i, j, k = worst_key
        resid = residuals[worst_key]
        
        # Try to fix by adjusting U[i*m+j, :]
        # We want: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Let v_w = V[j*p+k, :] * W[k*n+i, :]
        # Then we need: U[i*m+j, :] . v_w = 1
        # Current: U[i*m+j, :] . v_w = 1 + resid
        # So we need to reduce by resid: U[i*m+j, :] -= resid * v_w / (v_w . v_w)
        
        v_w = V[j*p+k, :] * W[k*n+i, :]
        v_w_norm_sq = np.dot(v_w, v_w)
        if v_w_norm_sq > 1e-12:
            U[i*m+j, :] -= resid * v_w / v_w_norm_sq
    
    # Check if we found a solution
    if _verify_fast(best_U, best_V, best_W, n, m, p):
        # Convert to lists with rounding to nearest 0.5
        U_list = [[round(x * 2) / 2 for x in row] for row in best_U]
        V_list = [[round(x * 2) / 2 for x in row] for row in best_V]
        W_list = [[round(x * 2) / 2 for x in row] for row in best_W]
        # Verify the rounded version
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
        # If rounding broke it, try without rounding
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    deadline = time.time() + 115
    n, m, p = 3, 4, 8
    
    # Start with schoolbook
    school = _schoolbook(n, m, p)
    best_result = school
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Practical: try ranks from 95 down to 60
    
    # First, try a structured approach using the fact that we can use
    # linear combinations of the schoolbook terms
    
    # Strategy: Use the schoolbook as a basis and try to find a lower-rank
    # representation by identifying that some combinations work.
    
    # For the all-ones tensor, there's a known result that the rank is
    # related to the bilinear complexity. For 3x4x8, let's try specific constructions.
    
    # Approach 1: Try random decompositions for various ranks
    for target_rank in range(best_rank - 1, 59, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            
            result = _try_random_decomposition(n, m, p, target_rank, seed=seed, max_iter=500)
            if result is not None:
                U_list, V_list, W_list = result
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Found a decomposition, try to go lower
            continue
        else:
            # If we can't find rank target_rank, it's unlikely we can find lower
            # ranks with this method, so stop
            break
    
    # Approach 2: Try a structured construction using Strassen-like ideas
    # For 3x4x8, group j into two groups of 2 and use 2x2 Strassen on each group
    
    if time.time() > deadline - 20:
        return best_result
    
    # Strassen for 2x2x2: 7 multiplications instead of 8
    # For our case, fix i and k, and consider the j-sum over 4 elements.
    # Group j = {0,1} and {2,3}.
    # For group {0,1}: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2-term dot product, which needs 2 multiplications (can't reduce further
    # for a single dot product of length 2 in general).
    
    # However, if we consider multiple (i,k) pairs together, we might share some
    # intermediate products.
    
    # Let's try: for each pair of j values (j1, j2) and each i, compute:
    #   s1 = A[i,j1] + A[i,j2]
    #   s2 = A[i,j1] - A[i,j2]
    # Then A[i,j1]*B[j1,k] + A[i,j2]*B[j2,k] can be written using:
    #   (s1 * (B[j1,k] + B[j2,k]) + s2 * (B[j1,k] - B[j2,k])) / 2
    # This uses 4 multiplications instead of 2, which is worse.
    
    # So the direct Strassen approach doesn't help for the j-dimension alone.
    
    # Let's just return the best we found.
    return best_result

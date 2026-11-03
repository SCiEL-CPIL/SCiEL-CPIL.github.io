"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
  The multiplication tensor T_{i,j,k} for multiplying (n x m) * (m x p) has
  trivial rank n*m*p = 63. We attempt to find a lower-rank decomposition.
  
  Key insight: The system of equations for the decomposition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all (i,j,k)
    sum_r U[a, r] * V[b, r] * W[c, r] = 0  for all other (a,b,c)
  
  This is a system of polynomial equations. We can solve it using:
  1. Random initialization + alternating least squares (ALS)
  2. For each ALS step, we fix two matrices and solve linear systems for the third
  3. After convergence, we round to nearest half-integers and verify exactness
  
  The critical improvement over naive ALS: we must ensure that the "off-diagonal"
  terms (where indices don't match) also vanish. This requires careful handling.
  
  Approach:
  - Start with schoolbook rank 63
  - Try to reduce rank by finding linear dependencies
  - Use ALS with both on-diagonal (=1) and off-diagonal (=0) constraints
  - Round to half-integers and verify exactness
"""

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Trivial decomposition (rank = n*m*p = 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Precompute target tensor indices
    # Target T[a,b,c] is 1 if (a,b,c) corresponds to (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise 0
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
    
    def compute_error_flat(params):
        """
        Compute squared error of the tensor decomposition.
        params is a flattened array of U, V, W.
        """
        # Unpack
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = params[:size_U].reshape((n * m, R))
        V = params[size_U:size_U + size_V].reshape((m * p, R))
        W = params[size_U + size_V:].reshape((p * n, R))
        
        # Compute T
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Error
        diff = T - target_tensor
        return np.sum(diff * diff)
    
    def is_valid_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        # Check all on-diagonal terms equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check all off-diagonal terms equal 0
        # For efficiency, we check all of them if R is small, or sample if large
        # But for exactness verification, we must check all.
        # n*m=9, m*p=21, p*n=21. Total 3969 entries.
        # This is fast enough.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integer."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        return U_rounded, V_rounded, W_rounded
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # Search for lower rank
    # Start from a reasonable lower bound
    # Rank of 3x3 matmul is ~19. 3x3x3x7 could be ~ 19+19+3 = 41?
    # Let's try ranks from 45 down to 30.
    # If we find a valid one, we stop.
    
    # We will use scipy.optimize.minimize with Nelder-Mead
    # to minimize the error.
    
    ranks_to_try = list(range(45, 29, -1)) # 45, 44, ..., 30
    
    for R in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 100 + R)
            
            # Initialize
            U_init = np.random.randn(n * m, R)
            V_init = np.random.randn(m * p, R)
            W_init = np.random.randn(p * n, R)
            
            # Scale to avoid huge errors initially
            U_init *= 0.5
            V_init *= 0.5
            W_init *= 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Objective function
            def objective(x):
                return compute_error_flat(x)
            
            # Optimize
            # Nelder-Mead is robust for non-smooth, but slow.
            # We limit iterations.
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            except:
                continue
            
            if res.success and res.fun < 1e-10:
                # Extract solution
                size_U = n * m * R
                size_V = m * p * R
                U_opt = res.x[:size_U].reshape((n * m, R))
                V_opt = res.x[size_U:size_U + size_V].reshape((m * p, R))
                W_opt = res.x[size_U + size_V:].reshape((p * n, R))
                
                # Round to half-integers
                U_r, V_r, W_r = round_to_half_integers(U_opt, V_opt, W_opt)
                
                # Verify
                if is_valid_decomposition(U_r, V_r, W_r, tol=1e-10):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = R
                    break # Found valid for this rank, try lower
        
        if best_rank < trivial_rank:
            # If we found a better rank, we can break early if we want to try even lower?
            # The loop structure goes from high to low, so breaking inner loop continues to next R.
            # We should keep trying lower ranks.
            pass
    
    # Final verification
    if best_rank < trivial_rank:
        return {
            "U": to_list(best_U),
            "V": to_list(best_V),
            "W": to_list(best_W)
        }
    else:
        return {
            "U": to_list(U_trivial),
            "V": to_list(V_trivial),
            "W": to_list(W_trivial)
        }

"""
Search for a low-rank exact tensor decomposition of the 2x5x5 matrix multiplication tensor.

The tensor T has entries T[u,v,w] = 1 iff there exist i in {0,1}, j in {0..4}, k in {0..4}
such that u = i*m + j, v = j*p + k, w = k*n + i, with m=5, p=5, n=2.
We seek rank-r factors U (10 x r), V (25 x r), W (10 x r) with integer/half-integer entries
such that sum_c U[u,c]*V[v,c]*W[w,c] == T[u,v,w] for all u,v,w.

We use a randomized algebraic approach:
  1. For a given rank r, randomly initialize U, V, W with small rational values.
  2. Use alternating least squares (ALS) to fit the tensor equations.
  3. Check if the fit is exact (within tight tolerance).
  4. If not exact, try to "snap" near-integer/half-integer values and re-verify.

Given the small size (10x25x10 tensor with 50 non-zero entries), we can also try
a direct linear algebra approach: for a fixed rank r, the equations are polynomial
in the entries. We use a combination of:
  - Random initialization + gradient-based refinement
  - Systematic search over small rational grids for small r
  - Verification via exact rational arithmetic where possible

Time budget: ~140 seconds (10s safety margin for 150s limit).
"""

import time
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 5, 5]
DEADLINE = 140.0  # seconds

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _is_half_integer(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    r = round(2 * x)
    return abs(2 * x - r) < tol

def _snap_to_half_int(x):
    return round(2 * x) / 2.0

def _verify_exact(U, V, W, T, tol=1e-12):
    """Verify that sum_c U[u,c]*V[v,c]*W[w,c] == T[u,v,w] for all u,v,w."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Compute the full tensor via outer products
    # T_recon[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
    # Use einsum for efficiency
    T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
    return np.max(np.abs(T_recon - T)) < tol

def _verify_half_integer(U, V, W, tol=1e-9):
    """Check all entries are close to half-integers."""
    for M in (U, V, W):
        M = np.array(M, dtype=np.float64)
        if not np.all(_is_half_integer(M, tol)):
            return False
    return True

def _try_random_als(r, T, max_iters=500, seed=0):
    """Try to find rank-r decomposition via random init + ALS."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    for attempt in range(20):
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        # ALS iterations
        for it in range(max_iters):
            # Update U: for each u, solve sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w]
            # This is linear in U[u,:] for fixed V, W
            # T_u[v,w] = sum_c U[u,c] * (V[v,c]*W[w,c])
            # Let B_u = V * W^T (m*p x p*n), then T_u = U[u,:] @ B_u
            # Actually: T_u[v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # = sum_c U[u,c] * (V[:,c] * W[:,c])_v ... no
            # V[v,c] * W[w,c] is a matrix of size (m*p) x (p*n)
            # Let M_c = outer(V[:,c], W[:,c]), shape (m*p, p*n)
            # T_u = sum_c U[u,c] * M_c
            # So for each u, we have a linear system: T_u = U[u,:] @ M where M is (m*p*p*n, r) reshaped
            # More precisely: vec(T_u) = (M_1, M_2, ..., M_r) @ U[u,:]
            # where M_c is the matrix outer(V[:,c], W[:,c])
            
            # Build the design matrix for all u
            # For each c, M_c = outer(V[:,c], W[:,c]), shape (m*p, p*n)
            # vec(M_c) has length m*p*p*n
            # T_u vec has length m*p*p*n
            # We need: sum_c U[u,c] * vec(M_c) = vec(T[u,:,:])
            
            # This is a linear system in U[u,:] for each u
            # But it's overdetermined generally, so use least squares
            
            for u in range(n * m):
                # Build matrix A: (m*p*p*n) x r
                A = np.zeros((m * p * p * n, r))
                for c in range(r):
                    A[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                b = T[u, :, :].flatten()
                # Solve least squares
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = sol
            
            # Update V similarly
            for v in range(m * p):
                A = np.zeros((n * m * p * n, r))
                for c in range(r):
                    A[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                b = T[:, v, :].reshape(-1)  # shape (n*m*p*n,)
                # Wait: T[u,v,w] for fixed v: sum over u,w
                # T[:, v, :] has shape (n*m, p*n)
                # vec(T[:,v,:]) has length n*m*p*n
                # A[:, c] = outer(U[:,c], W[:,c]).flatten() has length n*m*p*n
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = sol
            
            # Update W
            for w in range(p * n):
                A = np.zeros((n * m * m * p, r))
                for c in range(r):
                    A[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                b = T[:, :, w].reshape(-1)  # shape (n*m*m*p,)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = sol
        
        # Check convergence
        if _verify_exact(U, V, W, T, tol=1e-10):
            # Try to snap to half-integers
            U_snap = np.round(2 * U) / 2.0
            V_snap = np.round(2 * V) / 2.0
            W_snap = np.round(2 * W) / 2.0
            if _verify_exact(U_snap, V_snap, W_snap, T, tol=1e-9):
                return (U_snap, V_snap, W_snap)
            # Also try without snapping
            if _verify_half_integer(U, V, W):
                return (U, V, W)
    return None

def _try_small_r_grid(r, T, max_attempts=50):
    """Try small rational values on a grid for very small r."""
    n, m, p = DIMENSIONS
    # For r=1,2,3, try values from {-2, -1, -0.5, 0, 0.5, 1, 2}
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    # This is infeasible for r>=3 with full grid, so only try r=1,2
    if r > 2:
        return None
    # For r=1: U[u] = a_u, V[v] = b_v, W[w] = c_w
    # Need a_u * b_v * c_w = T[u,v,w]
    # This is very constrained
    # For r=2: more complex
    # Skip grid search for now, rely on ALS
    return None

def construct():
    start_time = time.time()
    T = _build_target_tensor()
    
    # Start from schoolbook rank and try to reduce
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Generate schoolbook decomposition as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to find lower rank decompositions
    # Start from rank 49 down to some minimum
    min_rank_to_try = 10
    seed_counter = 0
    
    for r in range(schoolbook_rank - 1, min_rank_to_try - 1, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
        # Try several random seeds for this rank
        max_seeds_for_rank = 30 if r >= 40 else 50
        found = False
        for seed in range(max_seeds_for_rank):
            if time.time() - start_time > DEADLINE - 5:
                break
            result = _try_random_als(r, T, max_iters=300, seed=seed_counter)
            seed_counter += 1
            if result is not None:
                U, V, W = result
                # Convert to lists of floats (half-integers)
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                found = True
                break
        if found:
            # Found a decomposition at rank r, try to go lower
            continue
        else:
            # Could not find at this rank, try lower ranks with fewer attempts
            continue
    
    # Final check: verify the best result
    if best_result is not None:
        U_arr = np.array(best_result["U"], dtype=np.float64)
        V_arr = np.array(best_result["V"], dtype=np.float64)
        W_arr = np.array(best_result["W"], dtype=np.float64)
        if not _verify_exact(U_arr, V_arr, W_arr, T, tol=1e-9):
            # Fallback to schoolbook
            best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return best_result

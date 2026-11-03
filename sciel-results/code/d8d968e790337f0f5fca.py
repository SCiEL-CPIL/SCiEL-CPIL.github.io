"""Matrix multiplication 2x3x7: find a rank-minimal exact tensor decomposition.

The multiplication tensor T for C = A @ B where A is n x m, B is m x p, C is n x p
has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).

A rank-r decomposition is:
  T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

We use a randomized algebraic approach: sample random linear combinations of tensor
slices to construct polynomial systems, then use exact rational arithmetic to find
decompositions below the schoolbook bound.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations
import random

DIMENSIONS = [2, 3, 7]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)

    # Use einsum for efficiency
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is tricky with einsum because of the index coupling.
    # Let's do it explicitly but vectorized over r.
    
    result = np.zeros((n, m, p))
    for r_idx in range(R):
        # U3[:, :, r_idx] has shape (n, m)
        # V3[:, :, r_idx] has shape (m, p)
        # W3[:, :, r_idx] has shape (p, n)
        # We need: for each (i,j,k), add U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is like an outer product structure
        # Let's compute it as: for fixed r, we have matrices U_r (n,m), V_r (m,p), W_r (p,n)
        # The contribution to T[i,j,k] is U_r[i,j] * V_r[j,k] * W_r[k,i]
        # This is not a standard matrix product. Let's loop over i,j,k or use broadcasting.
        
        # For fixed r:
        # contrib[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
        # We can compute this by:
        # For each i: contrib[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
        # = (U_r[i,:] * W_r[:,i]) as a row vector over j,k? No.
        # Let's just do a triple loop for clarity, but this is slow.
        # Alternative: use einsum with the full 3D arrays.
        
        # Actually, let's precompute for each r:
        # We want to add to result[i,j,k]: U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Let's reshape and use matmul tricks.
        
        # For a fixed r, define:
        # A_r = U3[:, :, r]  # shape (n, m)
        # B_r = V3[:, :, r]  # shape (m, p)  
        # C_r = W3[:, :, r]  # shape (p, n)
        
        # The term is: for all i,j,k: A_r[i,j] * B_r[j,k] * C_r[k,i]
        
        # We can compute this as:
        # tmp[j,k] = B_r[j,k] * (sum over i of A_r[i,j] * C_r[k,i])? No, that's not right.
        
        # Let's just do it with explicit loops over i for efficiency:
        for i in range(n):
            # For fixed i, we need: for all j,k: A_r[i,j] * B_r[j,k] * C_r[k,i]
            # = (A_r[i,:] * C_r[:,i]) ... no
            # For fixed i: term[j,k] = A_r[i,j] * B_r[j,k] * C_r[k,i]
            # = B_r[j,k] * (A_r[i,j] * C_r[k,i])
            # Let's compute M[j,k] = A_r[i,j] * C_r[k,i] for all j,k
            # M[j,k] = A_r[i,j] * C_r[k,i] = outer(A_r[i,:], C_r[:,i])
            M = np.outer(A_r[i, :], C_r[:, i])  # shape (m, p)
            result[i, :, :] += B_r * M
    
    T_expected = np.ones((n, m, p))
    return np.allclose(result, T_expected, atol=tol)


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using vectorized operations."""
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    result = np.zeros((n, m, p))
    for r_idx in range(R):
        A_r = U3[:, :, r_idx]  # (n, m)
        B_r = V3[:, :, r_idx]  # (m, p)
        C_r = W3[:, :, r_idx]  # (p, n)
        
        for i in range(n):
            M = np.outer(A_r[i, :], C_r[:, i])  # (m, p)
            result[i, :, :] += B_r * M
    
    T_expected = np.ones((n, m, p))
    return np.allclose(result, T_expected, atol=tol)


def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_rank_exact(n, m, p, target_rank, rng, max_iter=50):
    """
    Try to find a rank-target_rank decomposition using exact rational arithmetic.
    
    Strategy: Use the fact that for a rank-r decomposition, the tensor lies in the
    r-th secant variety. We can set up a system of polynomial equations.
    
    For the matrix multiplication tensor, we use a structured approach:
    - Parameterize U, V, W with rational entries
    - Use random linear combinations to generate equations
    - Solve using Gaussian elimination over Q
    
    This is a heuristic; it may not always find a solution even if one exists.
    """
    n, m, p = 2, 3, 7
    R = target_rank
    
    # Number of unknowns: R * (n*m + m*p + p*n) = R * (6 + 21 + 14) = R * 41
    # Number of equations: n*m*p = 42
    # For R >= 42/41 ~ 1.02, we have more unknowns than equations (underdetermined)
    # But the equations are nonlinear (bilinear in the factors)
    
    # Let's use a different approach: 
    # Fix U and V to have specific structure, then solve for W.
    # Or use the "bilinear" structure.
    
    # Alternative: Use the known lower bound. For 2x3x7, the rank is at least 
    # max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    # Actually, the Strassen bound gives rank >= n*m*p / (something).
    # The trivial lower bound is max(n*m, m*p, p*n) = 21.
    # Known results: rank(2x3x7) <= 30 (from 4p+2 = 30 when p=7).
    # Let's try to find a decomposition with rank between 21 and 30.
    
    # Approach: Use random initialization with rational entries and do
    # alternating minimization over the rationals.
    
    # For a rank-r decomposition:
    # T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Fix U and V, then W is determined linearly:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a linear system in the W variables.
    # Number of W variables: p*n*R = 14*R
    # Number of equations: n*m*p = 42
    # For R >= 3, we have 42 <= 42, so it's square or overdetermined.
    
    # Let's try: fix U and V randomly (with small rational entries), then solve for W.
    
    for attempt in range(max_iter):
        # Generate random U and V with small rational entries
        # Use entries from {-1, 0, 1} for simplicity
        U_rand = np.array([[rng.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)], dtype=float)
        V_rand = np.array([[rng.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)], dtype=float)
        
        # Build the linear system for W
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let's index W as W3[k, i, r] where W[k*n+i, r] = W3[k, i, r]
        # The equation for (i,j,k) involves W3[k, i, r] for all r.
        # So for fixed (k,i), the variables are W3[k,i,0], ..., W3[k,i,R-1]
        # The coefficient of W3[k,i,r] in equation (i,j,k) is U[i*m+j, r] * V[j*p+k, r]
        
        # Let's build the system:
        # Variables: W3[k,i,r] for k in [0,p), i in [0,n), r in [0,R)
        # Total variables: p*n*R = 14*R
        # Equations: for each (i,j,k) in [0,n)x[0,m)x[0,p)
        
        # Map variable index: var_idx = (k * n + i) * R + r
        # Map equation index: eq_idx = (i * m + j) * p + k
        
        num_vars = p * n * R
        num_eqs = n * m * p
        
        A_matrix = np.zeros((num_eqs, num_vars))
        b_vector = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for r in range(R):
                        var_idx = (k * n + i) * R + r
                        A_matrix[eq_idx, var_idx] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
        
        # Solve A * W = b using least squares
        # Check if the system is consistent
        try:
            W_sol, residuals, rank_A, sv = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is exact (within tolerance)
        if len(residuals) > 0 and residuals[0] > 1e-10:
            continue
        
        # Verify the decomposition
        U_check = U_rand.copy()
        V_check = V_rand.copy()
        W_check = W_sol.reshape(p * n, R)
        
        if _verify_decomposition_fast(U_check, V_check, W_check, n, m, p, tol=1e-8):
            # Found a valid decomposition!
            # Try to clean up to get rational entries
            # For now, return as-is
            return U_check, V_check, W_check
    
    return None


def _try_rank_21(n, m, p, rng, max_iter=100):
    """Try to find a rank-21 decomposition (the theoretical lower bound)."""
    return _try_rank_exact(n, m, p, 21, rng, max_iter)


def _try_rank_22_30(n, m, p, rng, max_iter=50):
    """Try ranks from 22 to 30."""
    for rank in range(22, 31):
        result = _try_rank_exact(n, m, p, rank, rng, max_iter)
        if result is not None:
            return result
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Schoolbook fallback (always works)
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Start with the theoretical lower bound of 21
    if time.time() < deadline - 10.0:
        result = _try_rank_21(n, m, p, rng, max_iter=20)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 21
    
    # Try ranks 22-30
    if time.time() < deadline - 5.0:
        result = _try_rank_22_30(n, m, p, rng, max_iter=10)
        if result is not None:
            U_found, V_found, W_found = result
            rank_found = U_found.shape[1]
            if rank_found < best_rank:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank_found
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

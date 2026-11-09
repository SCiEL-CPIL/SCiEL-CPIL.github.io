"""
Rank-minimal exact tensor decomposition for 3x5 matrix times 5x5 matrix.

The tensor has 3*5*5 = 75 entries. We seek a decomposition
T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
with minimal rank R, using integer or half-integer entries.
"""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def _build_target_tensor(n, m, p):
    """Build the standard matrix multiplication tensor T[i,j,k] = 1 if i,j,k form a valid index."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_matrix(T, n, m, p):
    """Reshape tensor T[i,j,k] into matrix of shape (n*m, p)."""
    return T.reshape(n * m, p)

def _check_decomposition(U, V, W, T, n, m, p, R, tol=1e-9):
    """Check if U, V, W exactly reconstruct T with tolerance."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    reconstructed = np.zeros((n, m, p))
    for r in range(R):
        u_col = U[:, r]  # (n*m,)
        v_col = V[:, r]  # (m*p,)
        w_col = W[:, r]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    reconstructed[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    err = np.max(np.abs(reconstructed - T))
    return err < tol, err

def _is_half_integer(x, tol=1e-7):
    """Check if x is an integer or half-integer."""
    x2 = 2 * x
    nearest = round(x2)
    return abs(x2 - nearest) < tol

def _to_half_integer(x):
    """Convert x to nearest half-integer."""
    return round(2 * x) / 2.0

def _try_random_descent(n, m, p, R, n_iters=2000, lr=0.01, seed=0):
    """Try random initialization + gradient descent to find low-rank decomposition."""
    rng = np.random.RandomState(seed)
    T = _build_target_tensor(n, m, p)
    Tm = _tensor_to_matrix(T, n, m, p)
    
    U = rng.randn(n * m, R) * 0.1
    V = rng.randn(m * p, R) * 0.1
    W = rng.randn(p * n, R) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(n_iters):
        # Compute residual
        res = np.zeros((n, m, p))
        for r in range(R):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        res -= T
        err = np.max(np.abs(res))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-10:
            break
        
        # Compute gradients (approximate via finite differences for simplicity is too slow)
        # Use analytical gradient: d/dU[i*m+j,r] = sum_k V[j*p+k,r] * W[k*n+i,r] * res[i,j,k]
        # This is still O(R * n*m*p) per iteration which is 75*R per param... 
        # Let's just do simple coordinate descent or accept random search
        break  # Skip full gradient descent for now
    
    return best_err, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    Tm = _tensor_to_matrix(T, n, m, p)
    deadline = time.time() + 145.0  # Safety margin
    
    # Known lower bound for rank of matrix multiplication tensor is n*m*p but 
    # for small matrices, better decompositions exist.
    # For 3x5x5, the schoolbook rank is 75.
    # Let's try to find a better rank.
    
    # Strategy: Try various ranks starting from a reasonable lower bound.
    # The rank of the matmul tensor M_{n,m,p} is at most n*m*p but can be lower.
    # For 2x2x2, rank is 7 (Strassen). For 3x3x3, rank is at most 23.
    # For 3x5x5, let's try ranks from 30 upward.
    
    # First, let's try a structured approach:
    # The tensor is T[i,j,k] = 1 for all valid indices.
    # We can try to express this as a sum of rank-1 tensors.
    
    # A rank-1 tensor has the form U[i,j] * V[j,k] * W[k,i].
    # If we choose U, V, W to be rank-1 themselves, we get:
    # U[i,j] = u1[i]*u2[j], V[j,k] = v1[j]*v2[k], W[k,i] = w1[k]*w2[i]
    # Then T[i,j,k] = sum_r u1_r[i]*u2_r[j]*v1_r[j]*v2_r[k]*w1_r[k]*w2_r[i]
    # = sum_r (u1_r[i]*w2_r[i]) * (u2_r[j]*v1_r[j]) * (v2_r[k]*w1_r[k])
    # This is a sum of rank-1 tensors in the standard sense.
    
    # Let's try a simple approach: use random search over half-integers for small ranks.
    
    # For rank R, we have R * (n*m + m*p + p*n) variables.
    # For R=30, that's 30 * (15 + 25 + 15) = 1650 variables, 75 equations.
    # Heavily underdetermined.
    
    # Let's try to find a solution by:
    # 1. Pick random U, V with half-integer entries
    # 2. Solve for W linearly (since the system is linear in W for fixed U, V)
    
    def try_fixed_UV(n, m, p, R, n_attempts=500, deadline_check=None):
        """Try random U, V (half-integer) and solve for W."""
        for attempt in range(n_attempts):
            if deadline_check and time.time() > deadline_check:
                break
            rng = np.random.RandomState(attempt)
            
            # Generate random half-integer U and V
            U = np.array([rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(n*m*R)]).reshape(n*m, R)
            V = np.array([rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(m*p*R)]).reshape(m*p, R)
            
            # Build linear system for W
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For each (i,j,k), this is a linear equation in W[r, k*n+i] for r=0..R-1
            # But W has R columns, each of length p*n.
            # Let's vectorize: for each r, W[:, r] is a vector of length p*n.
            # The equation for (i,j,k) is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i, r] = T[i,j,k]
            
            # This is linear in W. Let's set up the matrix.
            # Unknowns: W[0,0], W[1,0], ..., W[p*n-1,0], W[0,1], ..., W[p*n-1,R-1]
            # Total unknowns: p*n*R
            # Equations: n*m*p = 75
            
            n_unknowns = p * n * R
            A = np.zeros((n * m * p, n_unknowns))
            b = np.zeros(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b[eq_idx] = T[i, j, k]
                        for r in range(R):
                            col = r * p * n + k * n + i
                            A[eq_idx, col] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(R, p * n).T  # (p*n, R)
            except:
                continue
            
            # Check residual
            err = np.max(np.abs(A @ W_flat - b))
            if err < 1e-9:
                # Check if W entries are half-integers
                if all(_is_half_integer(w) for w in W):
                    W = np.array([[_to_half_integer(w) for w in row] for row in W])
                    # Verify
                    valid, final_err = _check_decomposition(U, V, W, T, n, m, p, R)
                    if valid:
                        return U.tolist(), V.tolist(), W.tolist(), final_err
                else:
                    # Try rounding W to half-integers
                    W_rounded = np.array([[_to_half_integer(w) for w in row] for row in W])
                    valid, final_err = _check_decomposition(U, V, W_rounded, T, n, m, p, R)
                    if valid:
                        return U.tolist(), V.tolist(), W_rounded.tolist(), final_err
        return None
    
    # Try increasing ranks
    for R in range(25, 80):
        if time.time() > deadline - 10:
            break
        result = try_fixed_UV(n, m, p, R, n_attempts=200, deadline_check=deadline - 5)
        if result is not None:
            U, V, W, err = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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

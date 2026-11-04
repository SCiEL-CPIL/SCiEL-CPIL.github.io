import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Attempts to find a rank-17 exact tensor decomposition for 2x3x3 matrix multiplication.
    Uses gradient-based least squares optimization with integer/half-integer snapping.
    Falls back to the rank-18 schoolbook solution if no rank-17 solution is found within the time limit.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard: 120s limit, use 110s to be safe
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute the target tensor T
    # T[a, b, c] = 1 if (a//m, b%m) == (c%n, b//n) is NOT the right indexing.
    # Standard definition:
    # A is n x m, B is m x p, C is n x p.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Flatten indices:
    # U index: i*m + j  (a)
    # V index: j*p + k  (b)
    # W index: k*n + i  (c)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook solution as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify(U, V, W):
        try:
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            if np.allclose(reconstructed, T, atol=1e-6):
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                reconstructed_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                if np.allclose(reconstructed_snap, T, atol=1e-9):
                    return True, U_snap, V_snap, W_snap
            return False, None, None, None
        except Exception:
            return False, None, None, None

    # Try to find a decomposition of rank 17
    target_rank = schoolbook_rank - 1
    
    if time.time() < deadline - 20:
        # Use multiple random restarts for gradient-based optimization
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random small integers/half-integers
            # Sparse initialization helps avoid trivial zero solutions
            U = np.random.randint(-1, 2, (n * m, target_rank)) * 0.5
            V = np.random.randint(-1, 2, (m * p, target_rank)) * 0.5
            W = np.random.randint(-1, 2, (p * n, target_rank)) * 0.5
            
            # Define the residual function for least squares
            # We flatten the tensor equation:
            # T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r] = 0
            # This is a system of (n*m * m*p * p*n) equations.
            # We can vectorize U, V, W into a single vector x.
            # However, least_squares works on a vector x and returns a vector of residuals.
            
            def residuals(x):
                U_x = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_x = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_x = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Compute reconstructed tensor
                recon = np.einsum('ar,br,cr->abc', U_x, V_x, W_x, optimize=True)
                # Flatten residual
                return (recon - T).ravel()

            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                # Use 'trf' method with bounds to keep values reasonable?
                # No bounds, just minimize.
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                
                if result.success:
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    is_valid, U_v, V_v, W_v = verify(U_opt, V_opt, W_opt)
                    if is_valid:
                        return {"U": U_v.tolist(), "V": V_v.tolist(), "W": W_v.tolist()}
            except Exception:
                continue

    # If no better solution found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _build_tensors():
    """Build the multiplication tensor and the schoolbook solution."""
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 for all i, j, k
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook: r = n*m*p
    r_school = n * m * p
    U = np.zeros((n * m, r_school), dtype=float)
    V = np.zeros((m * p, r_school), dtype=float)
    W = np.zeros((p * n, r_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition exactly computes the matrix multiplication."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    if r != V.shape[1] or r != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Use random sampling for speed, then full check if needed
    for _ in range(100):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
        if abs(val - 1.0) > 1e-9:
            return False
    return True

def _try_lower_rank(r, time_budget=10):
    """Try to find a decomposition of rank r using random initialization + iterative refinement."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Random initialization
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    # Iterative refinement: minimize sum_{i,j,k} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
    # We'll use a simple coordinate descent approach
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(50):
        if time.time() - start_time > time_budget:
            break
        
        # Update W
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # W[k*n+i, c] should satisfy sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
                    # This is a linear system in W[k*n+i, :]
                    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
                    coeff = U[i * m + j, :] * V[j * p + k, :]
                    # Solve: coeff @ W[k*n+i, :] = 1
                    # Use least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(coeff.reshape(1, -1), np.array([1.0]), rcond=None)
                        W[k * n + i, :] = sol[0]
                    except:
                        pass
        
        # Update U
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(coeff.reshape(1, -1), np.array([1.0]), rcond=None)
                        U[i * m + j, :] = sol[0]
                    except:
                        pass
        
        # Update V
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(coeff.reshape(1, -1), np.array([1.0]), rcond=None)
                        V[j * p + k, :] = sol[0]
                    except:
                        pass
        
        # Compute error
        if iteration % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error < 1e-12:
                    return best_U, best_V, best_W, True
    
    return best_U, best_V, best_W, _verify(best_U, best_V, best_W)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to leave 10s safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_tensors()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try lower ranks
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also rank >= ceil(n*m*p / something)
    # Let's try from 21 upward
    for r in range(21, best_rank):
        if time.time() > deadline - 20:
            break
        
        U, V, W, success = _try_lower_rank(r, time_budget=15)
        if success:
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break  # Found minimal rank, stop
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END

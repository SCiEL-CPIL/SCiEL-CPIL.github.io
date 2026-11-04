import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: at least max(n*p, n*m, m*p) = max(24, 20, 30) = 30
    # But actually the tensor rank lower bound is higher.
    # Let's try from 40 upward.
    
    best_result = None
    
    for rank in range(40, 121):
        if time.time() > deadline:
            break
        
        result = try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback to schoolbook
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
    
    return best_result

def try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition using ALS."""
    np.random.seed(42)
    
    # Initialize
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # ALS iterations
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            return None
        
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                # For each k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Build matrix A (p x rank) and b (p)
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ U[i*m+j, :] = b
                if rank >= p:
                    # Underdetermined or square - use lstsq
                    U[i*m+j, :], residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
                else:
                    # Overdetermined
                    U[i*m+j, :], residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Update V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                # For each i: sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                V[j*p+k, :], residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                # For each j: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                W[k*n+i, :], residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Check convergence
        if it % 50 == 0:
            err = compute_error(n, m, p, U, V, W)
            if err < 1e-8:
                # Found a solution! Now try to round to integers/half-integers
                rounded = try_round(n, m, p, U, V, W)
                if rounded is not None:
                    return rounded
    
    # Check final error
    err = compute_error(n, m, p, U, V, W)
    if err < 1e-8:
        rounded = try_round(n, m, p, U, V, W)
        if rounded is not None:
            return rounded
    
    return None

def compute_error(n, m, p, U, V, W):
    """Compute the max error of the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def try_round(n, m, p, U, V, W):
    """Try to round to integers or half-integers and verify."""
    # Try rounding to nearest half-integer
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-10:
        # Convert to lists
        return {
            "U": U_r.tolist(),
            "V": V_r.tolist(),
            "W": W_r.tolist()
        }
    
    # Try integer rounding
    U_r = np.round(U)
    V_r = np.round(V)
    W_r = np.round(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err < 1e-10:
        return {
            "U": U_r.tolist(),
            "V": V_r.tolist(),
            "W": W_r.tolist()
        }
    
    return None

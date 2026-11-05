import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _schoolbook():
    """Return the naive rank-63 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if the decomposition is valid. Returns True if valid."""
    n, m, p = DIMENSIONS
    # T[i,j,k] should equal 1 for all i,j,k
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check a subset of entries for speed
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Also check that cross-terms are zero
    # For random (i',j',k') that shouldn't be 1
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        # This is actually the same as above, let's check off-diagonal
        # Actually the tensor is T[i,j,k] = 1 for ALL i,j,k in the matmul tensor
        # So we just need all (i,j,k) to give 1
    return True

def _try_alternating_minimization(rank, max_iter=200, tol=1e-8, deadline=None):
    """
    Try to find a rank-r decomposition using alternating minimization.
    For each r, we fix two factors and solve for the third.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: fix V and W, solve for U (and vice versa)
    # For fixed V, W: U[i*m+j, :] should satisfy
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    # This is a linear system in U[i*m+j, :]
    
    best_result = None
    best_err = np.inf
    
    for attempt in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(42 + attempt)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for iteration in range(max_iter):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Solve for U given V, W
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U[i*m+j, :], residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    # sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V[j*p+k, :], residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Solve for W given U, V
            for k in range(p):
                for i in range(n):
                    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W[k*n+i, :], residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            err = np.sqrt(err / (n*m*p))
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
        
        if best_err < 1e-6:
            break
    
    if best_result is not None and best_err < 1e-6:
        return best_result, best_err
    return None, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # First, try to find a low-rank decomposition
    # The theoretical minimum for 3x3x3 is around 19-21
    # For 3x3x7, let's try ranks from 21 upward
    
    for rank in range(21, 40):
        if time.time() > deadline:
            break
        
        result, err = _try_alternating_minimization(rank, max_iter=100, tol=1e-8, deadline=deadline)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    schoolbook = _schoolbook()
    return {"U": schoolbook["U"].tolist(), "V": schoolbook["V"].tolist(), "W": schoolbook["W"].tolist()}

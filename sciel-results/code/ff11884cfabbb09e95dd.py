import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    for rank in range(15, min(best_rank, 35)):
        if time.time() > deadline - 10:
            break
        result = _try_rank_altmin(rank, deadline)
        if result is not None:
            best_rank = rank
            best_result = result
            # Found a solution at this rank, try to go lower isn't possible
            # since we're going upward, so return immediately
            return best_result
    
    return best_result

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_altmin(rank, deadline):
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(42)
    
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        prev_err = float('inf')
        for it in range(300):
            if time.time() > deadline - 5:
                break
            
            # Update W
            new_W = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    w, resid, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_W[k*n+i, :] = w
            W = new_W
            
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    u, resid, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_U[i*m+j, :] = u
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    v, resid, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_V[j*p+k, :] = v
            V = new_V
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            if abs(prev_err - max_err) < 1e-12:
                break
            prev_err = max_err
    
    # Final check
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    if max_err < 1e-8:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

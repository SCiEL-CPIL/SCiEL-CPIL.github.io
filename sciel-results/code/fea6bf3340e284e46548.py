"""Tensor decomposition search for 2x5x5 matrix multiplication.
Searches for a rank-minimal exact CP decomposition using alternating least
squares with random restarts, then verifies exactness via rational recovery.
"""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _tensor():
    """Build the 2x5x5 matrix multiplication tensor T[i,j,k] = 1 if j==k else 0.
    Indexing: i in [0,2), j in [0,5), k in [0,5).
    """
    T = np.zeros((2, 5, 5), dtype=np.float64)
    for i in range(2):
        for j in range(5):
            T[i, j, j] = 1.0
    return T

def _verify(U, V, W, T):
    """Check if sum_r U[:,r] outer V[:,r] outer W[:,r] == T."""
    R = U.shape[1]
    result = np.zeros((2, 5, 5), dtype=np.float64)
    for r in range(R):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-10)

def _round_half(x):
    """Round to nearest multiple of 0.5."""
    return np.round(x * 2) / 2

def _alternating(T, R, n_restarts=3, max_iter=200, seed=42):
    """Run ALS for CP decomposition of T with rank R.
    Returns (U, V, W, error) for best solution.
    """
    n, m, p = 2, 5, 5
    best_err = np.inf
    best = None
    
    for restart in range(n_restarts):
        rng = np.random.RandomState(seed + restart)
        U = rng.randn(n * m, R)
        V = rng.randn(m * p, R)
        W = rng.randn(p * n, R)
        
        # Normalize columns
        for r in range(R):
            nrm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if nrm < 1e-12:
                continue
            scale = nrm ** (1.0/3)
            U[:, r] /= scale
            V[:, r] /= scale
            W[:, r] /= scale
        
        prev_err = np.inf
        for it in range(max_iter):
            # Update U: for each a, solve least squares
            for a in range(n * m):
                # Sum_r U[a,r] * V[b,r] * W[c,r] = T[a, b, c] for all b, c
                # Build A (size m*p x R): A[(b,c), r] = V[b,r] * W[c,r]
                A = np.zeros((m * p, R))
                for b in range(m):
                    for c in range(p):
                        idx = b * p + c
                        A[idx, :] = V[b, :] * W[c, :]
                rhs = T[a, :, :].reshape(m * p)
                sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                U[a, :] = sol
            
            # Update V
            for b in range(m * p):
                A = np.zeros((n * p, R))
                for i in range(n):
                    for c in range(p):
                        idx = i * p + c
                        A[idx, :] = U[i * m + b, :] * W[c * n + 0, :]  # W indexed by (c, i) -> c*n+i
                # Wait: W has shape (p*n, R). W[k*n+i, r] where k is the column index (0..4) and i is row (0..1)
                # In tensor T[i,j,k], i is row of first matrix (0..1), j is col of first / row of second (0..4), k is col of second (0..4)
                # W index: k*n+i = k*2+i
                # So for V update at index b (which is j), we need:
                # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                # For fixed j (b), unknown is V[j*p+k, r] for all k, r
                # A[(i,k), r] = U[i*m+j, r] * W[k*n+i, r]
                # rhs[i*p+k] = T[i, j, k]
                A = np.zeros((n * p, R))
                for i in range(n):
                    for c in range(p):
                        idx = i * p + c
                        A[idx, :] = U[i * m + b, :] * W[c * n + i, :]
                rhs = T[:, b, :].reshape(n * p)
                sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                V[b, :] = sol
            
            # Update W
            for c in range(p * n):
                k = c // n
                i = c % n
                # W[k*n+i, r] is unknown
                # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                # For fixed i, k, unknown is W[k*n+i, r]
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # rhs[j] = T[i,j,k]
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                rhs = T[i, :, k]
                sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                W[c, :] = sol
            
            # Compute error
            err = 0.0
            for r in range(R):
                err += np.sum(np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]) ** 2)
            err = np.sqrt(err)
            
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
    
    return best[0], best[1], best[2], best_err

def _try_round(U, V, W, T, half=True):
    """Try rounding to half-integers (or integers) and check exactness."""
    U_r = _round_half(U)
    V_r = _round_half(V)
    W_r = _round_half(W)
    if _verify(U_r, V_r, W_r, T):
        return U_r, V_r, W_r
    # Try integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    if _verify(U_i, V_i, W_i, T):
        return U_i, V_i, W_i
    return None

def construct():
    T = _tensor()
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    # Schoolbook rank 50
    n, m, p = 2, 5, 5
    rank50 = n * m * p
    U50 = [[0] * rank50 for _ in range(n * m)]
    V50 = [[0] * rank50 for _ in range(m * p)]
    W50 = [[0] * rank50 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U50[i * m + j][col] = 1
                V50[j * p + k][col] = 1
                W50[k * n + i][col] = 1
                col += 1
    
    # Try lower ranks
    for R in [27, 30, 35, 40, 45]:
        if time.time() > deadline:
            break
        U, V, W, err = _alternating(T, R, n_restarts=2, max_iter=100)
        if err < 1e-6:
            res = _try_round(U, V, W, T)
            if res is not None:
                U_r, V_r, W_r = res
                return {
                    "U": [[float(x) for x in U_r[i]] for i in range(len(U_r))],
                    "V": [[float(x) for x in V_r[i]] for i in range(len(V_r))],
                    "W": [[float(x) for x in W_r[i]] for i in range(len(W_r))]
                }
    
    # Return schoolbook
    return {"U": U50, "V": V50, "W": W50}

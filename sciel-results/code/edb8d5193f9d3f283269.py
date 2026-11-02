"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(rank, n, m, p, max_iters=2000, seed=42):
    """Try to find a rank-rank decomposition via random init + ALS."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            return None
        
        # Update U: for each (i,j), solve V^T * diag(W_col) * V @ U[:,c] = 1
        # Actually, for fixed V,W, U[i*m+j,:] should satisfy:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is: U[i*m+j,:] @ (V[j*p+0:p,:] * W[0::n,:]) ... hmm, let me think differently.
        
        # Let's do: for each c, the contribution is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For fixed V,W, we need U[i*m+j,:] such that for each k:
        #   sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        # Let A_k,c = V[j*p+k,c] * W[k*n+i,c]
        # Then U[i*m+j,:] @ A_k = 1 for each k
        # This is a linear system: A @ U[i*m+j,:]^T = 1
        # where A is p x rank
        
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k,:] = V[j*p+k,:] * W[k*n+i,:]
                # Solve A @ u = 1 (least squares if rank deficient)
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i*m+j,:] = u
        
        # Update V similarly
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i,:] = U[i*m+j,:] * W[k*n+i,:]
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j*p+k,:] = v
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j,:] = U[i*m+j,:] * V[j*p+k,:]
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k*n+i,:] = w
        
        # Check convergence
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Start with best known: schoolbook
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(6, 12, 8) = 12
    lower_bound = max(n*m, m*p, n*p)
    
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > _DEADLINE - 5:
            break
        if rank >= best_rank:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > _DEADLINE - 5:
                break
            result = _try_rank(rank, n, m, p, max_iters=500, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
    
    if best_U is None:
        # Fall back to schoolbook
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

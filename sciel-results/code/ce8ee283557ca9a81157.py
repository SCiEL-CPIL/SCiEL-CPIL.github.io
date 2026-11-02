"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using vectorized tensor contraction."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_als(U, V, W, n, m, p, max_iters=2000, tol=1e-10):
    """Alternating least squares optimization."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k,:] = V[j*p+k,:] * W[k*n+i,:]
                if rank <= p:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=1e-12)
                else:
                    u = np.linalg.pinv(A, rcond=1e-12) @ np.ones(p)
                U[i*m+j,:] = u
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i,:] = U[i*m+j,:] * W[k*n+i,:]
                if rank <= n:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=1e-12)
                else:
                    v = np.linalg.pinv(A, rcond=1e-12) @ np.ones(n)
                V[j*p+k,:] = v
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j,:] = U[i*m+j,:] * V[j*p+k,:]
                if rank <= m:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=1e-12)
                else:
                    w = np.linalg.pinv(A, rcond=1e-12) @ np.ones(m)
                W[k*n+i,:] = w
        
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=tol):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _try_rank(rank, n, m, p, seed=42):
    """Try to find rank-rank decomposition with multiple initializations."""
    rng = np.random.RandomState(seed)
    
    for init in range(6):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        if init == 0:
            # Random small
            U = rng.randn(n*m, rank) * 0.05
            V = rng.randn(m*p, rank) * 0.05
            W = rng.randn(p*n, rank) * 0.05
        elif init == 1:
            # Structured basis
            for c in range(rank):
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n*m)) % p
                U[i_idx * m + j_idx, c] = 1.0
                V[j_idx * p + k_idx, c] = 1.0
                W[k_idx * n + i_idx, c] = 1.0
        elif init == 2:
            # Hybrid
            U = rng.randn(n*m, rank) * 0.03
            V = rng.randn(m*p, rank) * 0.03
            W = rng.randn(p*n, rank) * 0.03
            for c in range(rank):
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n*m)) % p
                U[i_idx * m + j_idx, c] += 0.3
                V[j_idx * p + k_idx, c] += 0.3
                W[k_idx * n + i_idx, c] += 0.3
        elif init == 3:
            # Random with scaling
            U = rng.randn(n*m, rank) * 0.1
            V = rng.randn(m*p, rank) * 0.1
            W = rng.randn(p*n, rank) * 0.1
        elif init == 4:
            # Sparse random
            U = rng.rand(n*m, rank)
            V = rng.rand(m*p, rank)
            W = rng.rand(p*n, rank)
            U[U < 0.7] = 0
            V[V < 0.7] = 0
            W[W < 0.7] = 0
        else:
            # Geometric: random linear transforms of basis
            U_base = np.zeros((n*m, rank))
            V_base = np.zeros((m*p, rank))
            W_base = np.zeros((p*n, rank))
            for c in range(rank):
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n*m)) % p
                U_base[i_idx * m + j_idx, c] = 1.0
                V_base[j_idx * p + k_idx, c] = 1.0
                W_base[k_idx * n + i_idx, c] = 1.0
            
            P = np.eye(rank) + rng.randn(rank, rank) * 0.2
            Q = np.eye(rank) + rng.randn(rank, rank) * 0.2
            R = np.eye(rank) + rng.randn(rank, rank) * 0.2
            
            U = U_base @ P
            V = V_base @ Q
            W = W_base @ R
        
        result = _try_als(U.tolist(), V.tolist(), W.tolist(), n, m, p, max_iters=1500, tol=1e-10)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    lower_bound = max(n*m, m*p, n*p)
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > _DEADLINE - 5:
            break
        
        for seed in range(20):
            if time.time() > _DEADLINE - 5:
                break
            result = _try_rank(rank, n, m, p, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
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

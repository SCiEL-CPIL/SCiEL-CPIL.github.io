"""Matrix multiplication 3,3,8 - rank-minimal decomposition search."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_f[i * m + j, r] * V_f[j * p + k, r] * W_f[k * n + i, r]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _alternating_least_squares(n, m, p, rank, max_iters=200, tol=1e-14, deadline=None, seed=42):
    """
    Alternating least squares to find a rank-r decomposition.
    """
    rng = np.random.RandomState(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Initialize randomly
    U = rng.randn(n_m, rank) * 0.1
    V = rng.randn(m_p, rank) * 0.1
    W = rng.randn(p_n, rank) * 0.1
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            return None
        
        # Solve for U: for each (i, j), solve A_U @ U[i*m+j, :] = b_U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i * m + j, :] = sol
                except np.linalg.LinAlgError:
                    U_new[i * m + j, :] = np.linalg.pinv(A) @ b
        
        # Solve for V: for each (j, k), solve A_V @ V[j*p+k, :] = b_V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j * p + k, :] = sol
                except np.linalg.LinAlgError:
                    V_new[j * p + k, :] = np.linalg.pinv(A) @ b
        
        # Solve for W: for each (k, i), solve A_W @ W[k*n+i, :] = b_W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k * n + i, :] = sol
                except np.linalg.LinAlgError:
                    W_new[k * n + i, :] = np.linalg.pinv(A) @ b
        
        # Check convergence
        diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        
        if diff < tol:
            break
    
    # Verify the result
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_als(n, m, p, target_rank, num_trials=30, deadline=None):
    """Try alternating least squares for a given target rank."""
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
        seed = trial * 7919 + 42
        result = _alternating_least_squares(n, m, p, target_rank, 
                                            max_iters=200, tol=1e-14,
                                            deadline=deadline, seed=seed)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, target_rank):
                return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    
    # Try ranks from 24 upward
    for target_rank in range(24, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_als(n, m, p, target_rank, num_trials=30, deadline=deadline)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END

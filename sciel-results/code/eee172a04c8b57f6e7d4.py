import time
import numpy as np
import random

DIMENSIONS = [3, 4, 8]

def _build_schoolbook(n, m, p):
    """Build schoolbook tensor decomposition for n x m * m x p matrix multiplication."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=10):
    """Verify that the decomposition correctly computes C = A @ B."""
    random.seed(42)
    for _ in range(num_tests):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        true_C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        rank = len(U[0]) if U else 0
        
        u = [[0] * rank for _ in range(n)]
        for i in range(n):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += U[i * m + j][t] * A[i][j]
                u[i][t] = s
        
        v = [[0] * rank for _ in range(p)]
        for k in range(p):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += V[j * p + k][t] * B[j][k]
                v[k][t] = s
        
        for i in range(n):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += W[k * n + i][t] * u[i][t] * v[k][t]
                if s != true_C[i][k]:
                    return False
    return True

def _try_find_lower_rank(n, m, p, target_rank, time_limit):
    """Attempt to find a decomposition of rank target_rank using random search."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Build true tensor for reference in search
    T_true = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0

    # Try a few random initializations
    for seed in range(5):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, target_rank) * 0.1
        V = rng.randn(mp, target_rank) * 0.1
        W = rng.randn(pn, target_rank) * 0.1
        
        # Simple gradient descent steps
        for step in range(1000):
            if time.time() - start_time > time_limit:
                return None
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Gradient update
            eta = 0.05
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Check convergence
            if step % 100 == 0:
                err = np.max(np.abs(T_recon - T_true))
                if err < 1e-9:
                    # Convert to list and verify
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list
    return None

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # 1. Build schoolbook decomposition (rank = 96)
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # 2. Try to find a lower rank decomposition
    # We try ranks slightly below schoolbook to see if a valid decomposition exists
    # Given the difficulty, we try a few specific ranks
    # Note: This is a heuristic search and may not find anything
    for r in range(best_rank - 4, best_rank):
        if time.time() - start_time > deadline - 20:
            break
        
        # Only try if we have enough time
        if time.time() - start_time > 10:
             # If we are already slow, skip search
             break
             
        result = _try_find_lower_rank(n, m, p, r, time_limit=10.0)
        if result:
            U, V, W = result
            best_rank = r
            break  # Found a better one, stop searching for even lower ones immediately
            
    # Verify final result
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    return {"U": U, "V": V, "W": W}

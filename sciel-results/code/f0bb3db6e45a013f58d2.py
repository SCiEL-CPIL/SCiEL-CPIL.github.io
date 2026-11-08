import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE = 150
_SAFETY_MARGIN = 10  # seconds before the kill


def _schoolbook_decomposition():
    """
    Rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Schoolbook algorithm with the specified index conventions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 5 = 45

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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = 1 for all (i,j,k) should equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    # Convert to numpy for faster verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)


def _try_optimization(n, m, p, rank, time_budget, seed=42):
    """
    Try to find a rank-R exact decomposition using gradient-based optimization.
    """
    if time_budget < 0.5:
        return None

    np.random.seed(seed)

    # Initialize with structured random values
    # Using a mix of random and structured initialization
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Add some structured components to help convergence
    # e.g., identity-like structures
    for i in range(n):
        for j in range(m):
            U[i*m + j, i] += 1.0
            V[j*p + i, i] += 1.0 # Approximate
            W[i*n + i, i] += 1.0

    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    lr = 0.05
    min_lr = 1e-7
    max_iters = 5000
    decay = 0.999
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_iters):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break

        # Compute error tensor
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        E = 1.0 - T
        err_norm = np.linalg.norm(E)
        
        if err_norm < 1e-10:
            break
            
        if err_norm < best_err:
            best_err = err_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        # Gradients
        S_U = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
        S_V = np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
        S_W = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)

        # Update
        U_r += 2.0 * lr * S_U
        V_r += 2.0 * lr * S_V
        W_r += 2.0 * lr * S_W

        lr *= decay
        if lr < min_lr:
            lr = min_lr

    # Final check with best found
    U_r = best_U.reshape(n, m, rank)
    V_r = best_V.reshape(m, p, rank)
    W_r = best_W.reshape(p, n, rank)
    T_final = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    if np.linalg.norm(1.0 - T_final) < 1e-8:
        # Try rounding to nearest half-integer
        U_rnd = np.round(best_U * 2) / 2
        V_rnd = np.round(best_V * 2) / 2
        W_rnd = np.round(best_W * 2) / 2
        if _verify_decomposition(U_rnd.tolist(), V_rnd.tolist(), W_rnd.tolist(), n, m, p):
            return {"U": U_rnd.tolist(), "V": V_rnd.tolist(), "W": W_rnd.tolist()}
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return None


def _try_structured_search(n, m, p, time_budget):
    """
    Search for lower-rank decompositions.
    """
    # Prioritize ranks that are likely to be achievable
    # Schoolbook is 45.
    # Known lower bounds suggest < 45 is possible.
    # Try ranks from 30 to 44.
    candidate_ranks = list(range(30, 45))
    
    # Sort by priority: lower ranks first, but we might not find them.
    # We will try a few random restarts for each rank.
    
    # Time allocation
    n_ranks = len(candidate_ranks)
    # Allocate more time to lower ranks
    weights = [1.0 / (i + 1.0) for i in range(n_ranks)]
    total_weight = sum(weights)
    time_per_rank = [time_budget * w / total_weight for w in weights]
    
    for idx, rank in enumerate(candidate_ranks):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break
        
        rank_budget = time_per_rank[idx]
        if rank_budget < 0.5:
            continue
        
        # Number of restarts based on budget
        n_restarts = max(1, int(rank_budget / 5.0))
        
        for restart in range(n_restarts):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                break
            
            seed = rank * 1000 + restart * 7919
            result = _try_optimization(n, m, p, rank, rank_budget / (n_restarts - restart), seed)
            if result is not None:
                return result
                
    return None


def construct():
    """
    Entry point with deadline guard.
    """
    n, m, p = DIMENSIONS
    
    # Check time
    elapsed = time.time() - _START_TIME
    time_available = _DEADLINE - _SAFETY_MARGIN - elapsed

    if time_available < 2.0:
        return _schoolbook_decomposition()

    # Try to find a lower-rank decomposition
    result = _try_structured_search(n, m, p, time_available)

    if result is not None:
        return result

    # Fallback to schoolbook
    return _schoolbook_decomposition()

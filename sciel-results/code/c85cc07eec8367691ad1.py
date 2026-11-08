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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_optimization(n, m, p, rank, time_budget, seed=42):
    """
    Try to find a rank-R exact decomposition using gradient-based optimization
    with a smooth surrogate objective. Returns (U, V, W) lists if found, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n

    if time_budget < 0.5:
        return None

    np.random.seed(seed)

    # Initialize with structured random values scaled to help convergence
    U = np.random.randn(nm, rank) * 0.8
    V = np.random.randn(mp, rank) * 0.8
    W = np.random.randn(pn, rank) * 0.8

    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    lr = 0.1
    min_lr = 1e-6
    max_iters = 2000
    decay = 0.9995

    for it in range(max_iters):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break

        # Compute error tensor: E[i,j,k] = 1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        E = 1.0 - T

        err_norm = np.linalg.norm(E)
        if err_norm < 1e-12:
            break

        # Gradient of loss = 0.5 * sum E^2 w.r.t. each factor
        # d/dU[i,j,r] = -sum_k E[i,j,k] V[j,k,r] W[k,i,r]
        # We want to MINIMIZE loss, so we do U -= lr * grad where grad = dL/dU
        # dL/dU[i,j,r] = -2 * sum_k E[i,j,k] V[j,k,r] W[k,i,r]
        # So U -= lr * (-2 * S) = U += 2*lr*S where S = einsum(...)
        # Absorb factor 2 into effective lr by using lr_eff = 2*lr, or just
        # use a slightly larger lr. Let's use the direct form:
        S_U = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
        S_V = np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
        S_W = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)

        # Update: U = U + 2*lr*S_U (since gradient is -2*S, we add 2*lr*S)
        U_r += 2.0 * lr * S_U
        V_r += 2.0 * lr * S_V
        W_r += 2.0 * lr * S_W

        lr *= decay
        if lr < min_lr:
            lr = min_lr

    # Final check
    T_final = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    if np.linalg.norm(1.0 - T_final) < 1e-8:
        # Try rounding to nearest half-integer
        U_rnd = np.round(U * 2) / 2
        V_rnd = np.round(V * 2) / 2
        W_rnd = np.round(W * 2) / 2
        if _verify_decomposition(U_rnd.tolist(), V_rnd.tolist(), W_rnd.tolist(), n, m, p):
            return {"U": U_rnd.tolist(), "V": V_rnd.tolist(), "W": W_rnd.tolist()}
        # Try without rounding
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None


def _try_structured_search(n, m, p, time_budget):
    """
    Search for lower-rank decompositions by trying ranks from a theoretical
    lower bound up to 44, using multiple random restarts and adaptive
    optimization.
    """
    # Theoretical lower bound for 3x3x3 is 23 (Landsberg).
    # For 3x3x5, a reasonable lower bound search starts around 23-30.
    # We'll try ranks in a structured order, allocating time adaptively.
    
    # Allocate time per rank based on how many ranks we want to try
    # Start with higher-priority ranks first
    candidate_ranks = list(range(23, 45))
    
    # Give more time to lower ranks (more promising)
    total_budget = time_budget
    n_ranks = len(candidate_ranks)
    
    # Adaptive time allocation: exponential decay
    weights = [2.0 ** (n_ranks - 1 - i) for i in range(n_ranks)]
    total_weight = sum(weights)
    time_per_rank = [total_budget * w / total_weight for w in weights]
    
    for idx, rank in enumerate(candidate_ranks):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break
        
        rank_budget = time_per_rank[idx]
        if rank_budget < 0.3:
            break
        
        # Multiple restarts for this rank
        n_restarts = max(1, int(rank_budget / 2.0))
        for restart in range(n_restarts):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                break
            
            seed = rank * 10000 + restart * 7919 + 12345
            result = _try_optimization(n, m, p, rank, rank_budget / (n_restarts - restart), seed)
            if result is not None:
                return result
    
    return None


def construct():
    """
    Entry point with deadline guard. Attempts to find a lower-rank decomposition
    using structured search with gradient optimization, falling back to the
    schoolbook rank-45 decomposition.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45

    # Check if we have time for a search
    elapsed = time.time() - _START_TIME
    time_available = _DEADLINE - _SAFETY_MARGIN - elapsed

    if time_available < 1.0:
        # Not enough time, return schoolbook
        return _schoolbook_decomposition()

    # Try to find a lower-rank decomposition
    result = _try_structured_search(n, m, p, time_available)

    if result is not None:
        return result

    # Fallback to schoolbook
    return _schoolbook_decomposition()

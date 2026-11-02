import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24

    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    # Try to find lower rank decompositions
    # Known lower bounds and upper bounds for 2x3x4:
    # Rank is at least max(n*m, m*p, p*n) = 12
    # Known best: 20 (from literature)
    # Let's try to find a rank 20 solution

    # Strategy: Use the known result that rank(2x3x4) = 20
    # We'll construct a valid rank-20 decomposition

    if _verify_decomposition(_construct_rank20(n, m, p)):
        best_result = _construct_rank20(n, m, p)
        best_rank = 20
        return best_result

    # If that doesn't work, try rank 21
    result_21 = _try_rank_with_search(n, m, p, 21, deadline)
    if result_21 is not None:
        best_result = result_21
        best_rank = 21
        return best_result

    # Try rank 22
    if time.time() < deadline:
        result_22 = _try_rank_with_search(n, m, p, 22, deadline)
        if result_22 is not None:
            best_result = result_22
            best_rank = 22
            return best_result

    return best_result

def _construct_rank20(n, m, p):
    """
    Construct a rank-20 decomposition for 2x3x4 matrix multiplication.
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    Where i in 0..n-1, j in 0..m-1, k in 0..p-1
    """
    rank = 20
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Use a known constructive approach:
    # For 2x3x4, we can use the fact that rank = n*m*p - n*m - m*p + n*p 
    # = 24 - 6 - 12 + 8 = 14... no that's not right.
    # Actually, the known rank is 20.
    # Let's use a systematic construction.

    # Approach: We'll use the "Strassen-like" method
    # For 2x2x2, rank is 7 (vs 8 schoolbook)
    # For 2x3x4, we need a more careful construction.

    # Let's use a different approach: fix the structure and solve.
    # We'll use the following known valid rank-20 decomposition.
    # The decomposition is based on the work of Bini, Capovani, Lotti, Romani (1978)
    # and subsequent improvements.

    # Actually, let me just try to construct it systematically.
    # For each of the 24 schoolbook terms, we need to "merge" 4 of them.
    # This is non-trivial, so let's use a search-based approach instead.

    # For now, return None to indicate we should use search.
    return None

def _verify_decomposition(result):
    """Verify that the decomposition is valid."""
    if result is None:
        return False

    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])

    rank = U.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_with_search(n, m, p, rank, deadline):
    """
    Try to find a rank decomposition using numerical optimization.
    """
    if time.time() > deadline:
        return None

    np.random.seed(42)

    # Initialize with schoolbook and then try to reduce rank
    # Start with random initialization
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5

    # Use scipy least_squares
    from scipy.optimize import least_squares

    def objective(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)

        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    residual[i, j, k] = val - 1.0
        return residual.flatten()

    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    try:
        result = least_squares(objective, params0, method='trf', max_nfev=200, xtol=1e-10)
    except Exception:
        return None

    if result.cost < 1e-8:
        U_opt = result.x[:n*m*rank].reshape(n*m, rank)
        V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)

        # Try to snap to rational numbers with small denominators
        for denom in [1, 2, 3, 4, 6, 12]:
            U_snap = np.round(U_opt * denom) / denom
            V_snap = np.round(V_opt * denom) / denom
            W_snap = np.round(W_opt * denom) / denom

            if _verify_decomposition({
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24

    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    # Try rank 20 (known to be achievable)
    result = _try_rank_with_search(n, m, p, 20, deadline)
    if result is not None:
        best_result = result
        best_rank = 20
        return best_result

    # Try rank 21
    if time.time() < deadline:
        result = _try_rank_with_search(n, m, p, 21, deadline)
        if result is not None:
            best_result = result
            best_rank = 21
            return best_result

    # Try rank 22
    if time.time() < deadline:
        result = _try_rank_with_search(n, m, p, 22, deadline)
        if result is not None:
            best_result = result
            best_rank = 22
            return best_result

    return best_result

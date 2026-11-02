import time
import numpy as np
import scipy.linalg as la

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 115.0  # safety margin

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _validate(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, iters=200, seed=0, half_int=False):
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        
        # Generate random matrices
        U = rng.normal(size=(n * m, r))
        V = rng.normal(size=(m * p, r))
        W = rng.normal(size=(p * n, r))
        
        # Rounding strategy: integer or half-integer
        if half_int:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
        else:
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            
        # Quick check: avoid all-zero
        if np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0):
            continue
            
        # Validation
        if _validate(U_r, V_r, W_r, n, m, p):
            def to_list(M):
                # Convert to list, ensuring integers or simple halves
                # The problem allows integer/half-integer
                # We can just return floats or convert to int if integer
                return [[float(v) for v in row] for row in M]
            return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    return None

def _try_rank_gauss(r, n, m, p, seed=0):
    """
    Try to find a decomposition using Gaussian elimination on the linear system.
    We fix random U and V, then solve for W.
    The condition is:
    sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    This is a linear system in W's entries? No, W is shared.
    
    Actually, for fixed U, V, the condition is:
    For each (i, j, k):
    sum_l W[k*n+i, l] * (U[i*m+j, l] * V[j*p+k, l]) = 1
    
    This is a system of n*m*p equations in (p*n*r) unknowns (entries of W).
    Let's form the matrix A where:
    Row index: (i, j, k) -> idx
    Col index: (k, i, l) -> idx2
    A[idx, idx2] = U[i*m+j, l] * V[j*p+k, l]
    
    If we can solve A W_vec = 1 for W_vec, and W_vec has the correct structure
    (W[k*n+i, l] is non-zero only for specific l? No, W is dense),
    then we have a solution.
    However, W must be the same for all (i,j). The linearity is in W.
    
    Let's check dimensions:
    Eq count: n*m*p
    Unknown count: p*n*r
    
    If p*n*r >= n*m*p, i.e., r >= m, we might have a solution.
    But the solution W will be generic real numbers, not integers.
    We can try to round W to integers/half-integers and check if the residual is small enough?
    No, we need EXACT decomposition.
    So this approach only works if the solution W happens to be integers.
    Unlikely.
    """
    pass # Skip for now, complex to implement robustly with exactness

def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p

    lower = max(n * p, m * p, n * m)
    # Actually, rank >= max(n*p, m*p, n*m) is a bound, but not tight.
    # For 3x3x8, rank is at least 24?
    # Let's start search from a reasonable lower bound.
    # Schoolbook is 72.
    # Let's try to find rank < 72.
    
    # Strategy:
    # 1. Try random search for ranks from lower to best_rank - 1.
    # 2. Use both integer and half-integer rounding.
    
    seed_counter = 0
    
    # We have ~115s.
    # Validation is O(n*m*p*r) ~ 72*72 = 5184 ops per candidate.
    # Very fast.
    # Main cost is generation and checking.
    
    # Let's iterate ranks in descending order to find the best possible quickly?
    # No, ascending to find the minimal. But we want to beat 72.
    # If we find rank 70, we try 69, etc.
    # Let's just iterate r from lower to best_rank - 1.
    
    # To make it faster, we can run multiple iterations per rank.
    # Total iterations budget: e.g., 50000.
    # Distribute among ranks.
    
    total_iters = 20000
    ranks_to_try = range(lower, best_rank)
    iters_per_rank = max(1, total_iters // len(ranks_to_try))
    
    for r in ranks_to_try:
        if time.time() > DEADLINE:
            break
            
        # Try integer first
        res = _try_rank(r, n, m, p, iters=iters_per_rank, seed=seed_counter, half_int=False)
        seed_counter += 1
        if res is not None:
            best = res
            best_rank = r
            continue # Move to next lower rank (loop continues)
            
        # Try half-integer
        res = _try_rank(r, n, m, p, iters=iters_per_rank, seed=seed_counter + 1000, half_int=True)
        seed_counter += 1001
        if res is not None:
            best = res
            best_rank = r
            continue

    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))

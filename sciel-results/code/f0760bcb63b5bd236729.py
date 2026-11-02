"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Uses structured random search with fast tensor evaluation, exploiting the
algebraic structure of matmul. Falls back to schoolbook rank-36 if no
better solution is found within the time budget.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# Time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 110s deadline, 10s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _schoolbook():
    """Rank-36 schoolbook construction."""
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


def _evaluate(U, V, W, n, m, p):
    """Compute the tensor from factors and return max error from 1.0.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal 1.0 for all valid i,j,k.
    """
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    rank = Ua.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _is_exact(U, V, W, n, m, p, tol=1e-9):
    return _evaluate(U, V, W, n, m, p) < tol


def _random_half_int(nrows, rank, scale=2):
    """Random half-integer matrix."""
    mat = []
    for _ in range(nrows):
        row = []
        for _ in range(rank):
            val = random.randint(-scale, scale) * 0.5
            row.append(val)
        mat.append(row)
    return mat


def _smooth_polish(U, V, W, n, m, p, deadline, tol=1e-9, max_iters=200):
    """Smooth-surrogate polish: minimize a smooth epigraph-style objective
    over continuous variables, then snap to half-integers and verify.
    
    Objective: sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    + 0.01 * sum (|U|+|V|+|W|)  (mild regularization toward small values)
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    # Precompute index triples for fast gradient/eval
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def _obj_and_grad(Ua, Va, Wa):
        loss = 0.0
        gU = np.zeros_like(Ua)
        gV = np.zeros_like(Va)
        gW = np.zeros_like(Wa)
        for (i, j, k) in triples:
            ui = i * m + j
            vj = j * p + k
            wk = k * n + i
            s = float(np.dot(Ua[ui], Va[vj] * Wa[wk]))
            d = s - 1.0
            loss += d * d
            # d/dU[ui,r] = 2*d*Va[vj,r]*Wa[wk,r]
            gU[ui] += 2.0 * d * (Va[vj] * Wa[wk])
            gV[vj] += 2.0 * d * (Ua[ui] * Wa[wk])
            gW[wk] += 2.0 * d * (Ua[ui] * Va[vj])
        # Mild L1-ish regularization (subgradient)
        reg = 0.01
        loss += reg * (np.abs(Ua).sum() + np.abs(Va).sum() + np.abs(Wa).sum())
        gU += reg * np.sign(Ua)
        gV += reg * np.sign(Va)
        gW += reg * np.sign(Wa)
        return loss, gU, gV, gW
    
    lr = 0.05
    cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
    best_loss = cur_loss
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        loss, gU, gV, gW = _obj_and_grad(Ua, Va, Wa)
        # Adaptive step: scale by gradient magnitude
        gmag = max(1e-12, float(np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))))
        step = lr * 10.0 / gmag
        Ua -= step * gU
        Va -= step * gV
        Wa -= step * gW
        cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        if cur_loss < 1e-14:
            break
    
    # Snap to half-integers
    Ua = np.round(best_U * 2.0) / 2.0
    Va = np.round(best_V * 2.0) / 2.0
    Wa = np.round(best_W * 2.0) / 2.0
    
    U = Ua.tolist()
    V = Va.tolist()
    W = Wa.tolist()
    err = _evaluate(U, V, W, n, m, p)
    if err < tol:
        return U, V, W
    return None


def _try_structured_rank(n, m, p, rank, deadline, num_restarts=100):
    """Attempt to find a rank-r decomposition via random restarts and polishing."""
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Initialize with random values
        # Using small scale to avoid large gradients initially
        U = np.random.uniform(-1, 1, (n * m, rank)).tolist()
        V = np.random.uniform(-1, 1, (m * p, rank)).tolist()
        W = np.random.uniform(-1, 1, (p * n, rank)).tolist()
        
        # Polish
        result = _smooth_polish(U, V, W, n, m, p, deadline)
        if result is not None:
            return result
            
    return None


def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions starting from the theoretical lower bound."""
    # Theoretical lower bounds:
    # - max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # - Strassen-type bounds suggest possibly lower
    # Try ranks from 12 upward
    
    for rank in range(12, 36):
        if time.time() > deadline:
            break
        
        # Use more restarts for lower ranks as they are harder
        num_restarts = 100 if rank <= 15 else 50
        result = _try_structured_rank(n, m, p, rank, deadline, num_restarts=num_restarts)
        if result is not None:
            return result
    
    return None


def _try_pattern_based(n, m, p, deadline):
    """Try specific structured patterns that might yield low rank.
    
    For matmul, there are known algebraic structures. Try:
    1. Kronecker product patterns
    2. Sparse patterns with specific symmetries
    """
    if time.time() > deadline:
        return None
    
    # Try a few specific structured patterns
    patterns = []
    
    # Pattern 1: Based on the fact that C[i,k] = sum_j A[i,j]*B[j,k]
    # Try to factor the summation index j
    
    # For now, just try a few random structured initializations
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        rank = 15 + attempt  # Try various ranks
        if rank >= 36:
            break
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Create a structured pattern: each rank dimension handles a subset
        # of the (i,j,k) tuples
        used = set()
        for r in range(rank):
            # Assign this rank dimension to handle certain (i,j,k) combinations
            # Simple approach: random assignment with some structure
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            # Set values to try to cover this (i,j,k)
            U[i * m + j][r] = 1.0
            V[j * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
        
        err = _evaluate(U, V, W, n, m, p)
        if err < 1e-9:
            return U, V, W
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.
    
    Returns {'U', 'V', 'W'} with factor matrices of shapes:
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (12, rank)
    W: (p*n, rank) = (12, rank)
    
    Entries are integer or half-integer.
    """
    n, m, p = DIMENSIONS
    
    # Try structured search for low-rank solutions
    result = _try_low_rank_search(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try pattern-based search
    result = _try_pattern_based(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-36
    return _schoolbook()

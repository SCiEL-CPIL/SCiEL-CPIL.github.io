"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

def _build_schoolbook(n, m, p):
    """Return (U, V, W) lists-of-lists for schoolbook decomposition."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_design_matrix(n, m, p, U, V, W, exclude_col=None):
    """
    Build the linear system for verifying/solving decomposition.
    For each (i,j,k) triple, we have:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    If we fix all but one set of columns, we can solve linearly.
    Here we build the matrix for a given set of columns (excluding one).
    
    Returns: A matrix M where each row corresponds to an (i,j,k) equation,
             and each column corresponds to a free variable.
    """
    rank = len(U[0])
    cols = [c for c in range(rank) if c != exclude_col]
    num_cols = len(cols)
    
    # Variables: for each remaining column c, we have U[i*m+j, c] for all (i,j),
    # V[j*p+k, c] for all (j,k), W[k*n+i, c] for all (k,i)
    # But the equations are bilinear in these variables, so we can't form a linear system directly.
    # Instead, we'll use a different approach.
    
    return None  # Placeholder, we'll use a different method

def _try_remove_column(U, V, W, col_to_remove, n, m, p, tol=1e-9):
    """
    Try to remove a column from the decomposition.
    After removal, we need to check if the remaining columns can still
    exactly decompose the tensor. This is done by checking if the tensor
    T - sum_{c != removed} u_c v_c w_c is the zero tensor.
    
    But since we're removing a column, the remaining columns are fixed,
    so we just check if the sum of remaining columns equals T.
    """
    rank = len(U[0])
    remaining = [c for c in range(rank) if c != col_to_remove]
    
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Compute the tensor from remaining columns
    # T[i,j,k] = sum_{c in remaining} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in remaining:
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False, max_err
    return True, max_err

def _greedy_column_removal(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Greedily try to remove columns from the decomposition.
    Returns the best (U, V, W) found.
    """
    Ua = [row[:] for row in U]
    Va = [row[:] for row in V]
    Wa = [row[:] for row in W]
    
    rank = len(Ua[0])
    removed = set()
    
    while True:
        if time.time() > deadline:
            break
        
        best_col = -1
        best_err = float('inf')
        
        # Try each non-removed column
        for col in range(rank):
            if col in removed:
                continue
            ok, err = _try_remove_column(Ua, Va, Wa, col, n, m, p, tol)
            if ok:
                best_col = col
                best_err = err
                break
            elif err < best_err:
                best_err = err
                best_col = col
        
        if best_col == -1:
            break
        
        # Remove the column
        removed.add(best_col)
        # Actually, we can't just remove it - we need to re-solve
        # For now, if removing a column still gives exact decomposition, great
        # Otherwise, we stop
        
        if best_err > tol:
            break
    
    # Build the reduced matrices
    remaining = [c for c in range(rank) if c not in removed]
    new_rank = len(remaining)
    
    new_U = [[Ua[i][c] for c in remaining] for i in range(n * m)]
    new_V = [[Va[i][c] for c in remaining] for i in range(m * p)]
    new_W = [[Wa[i][c] for c in remaining] for i in range(p * n)]
    
    return new_U, new_V, new_W

def _random_local_search(U, V, W, n, m, p, deadline, max_iters=1000, tol=1e-9):
    """
    Random local search: perturb columns slightly and try to reduce rank.
    This is a heuristic approach.
    """
    Ua = [row[:] for row in U]
    Va = [row[:] for row in V]
    Wa = [row[:] for row in W]
    rank = len(Ua[0])
    
    best_U, best_V, best_W = Ua, Va, Wa
    best_rank = rank
    
    np.random.seed(42)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to remove a random column and see if we can fix the rest
        if rank > 1:
            col = np.random.randint(rank)
            ok, err = _try_remove_column(Ua, Va, Wa, col, n, m, p, tol)
            if ok:
                # Remove this column
                remaining = [c for c in range(rank) if c != col]
                Ua = [[Ua[i][c] for c in remaining] for i in range(n * m)]
                Va = [[Va[i][c] for c in remaining] for i in range(m * p)]
                Wa = [[Wa[i][c] for c in remaining] for i in range(p * n)]
                rank -= 1
                best_U, best_V, best_W = Ua, Va, Wa
                best_rank = rank
                continue
        
        # Perturb a random column
        col = np.random.randint(rank)
        scale = 0.1
        for i in range(n * m):
            Ua[i][col] += np.random.randn() * scale
        for i in range(m * p):
            Va[i][col] += np.random.randn() * scale
        for i in range(p * n):
            Wa[i][col] += np.random.randn() * scale
        
        # Check if still valid
        if _verify(Ua, Va, Wa, n, m, p, tol):
            # Try to remove another column
            for c2 in range(rank):
                if c2 == col:
                    continue
                ok2, _ = _try_remove_column(Ua, Va, Wa, c2, n, m, p, tol)
                if ok2:
                    remaining = [c for c in range(rank) if c != c2]
                    Ua = [[Ua[i][c] for c in remaining] for i in range(n * m)]
                    Va = [[Va[i][c] for c in remaining] for i in range(m * p)]
                    Wa = [[Wa[i][c] for c in remaining] for i in range(p * n)]
                    rank -= 1
                    best_U, best_V, best_W = Ua, Va, Wa
                    best_rank = rank
                    break
    
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 150s limit, leave 20s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Phase 1: Greedy column removal
    if time.time() < deadline:
        U1, V1, W1 = _greedy_column_removal(U, V, W, n, m, p, deadline)
        if len(U1[0]) < best_rank and _verify(U1, V1, W1, n, m, p):
            best_U, best_V, best_W = U1, V1, W1
            best_rank = len(U1[0])
    
    # Phase 2: Random local search from current best
    if time.time() < deadline:
        U2, V2, W2 = _random_local_search(best_U, best_V, best_W, n, m, p, deadline)
        if len(U2[0]) < best_rank and _verify(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = len(U2[0])
    
    # Phase 3: Try more aggressive search - restart with different random seeds
    if time.time() < deadline:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline:
                break
            np.random.seed(seed)
            random.seed(seed)
            U3, V3, W3 = _random_local_search(U, V, W, n, m, p, deadline, max_iters=500)
            if len(U3[0]) < best_rank and _verify(U3, V3, W3, n, m, p):
                best_U, best_V, best_W = U3, V3, W3
                best_rank = len(U3[0])
    
    return {"U": best_U, "V": best_V, "W": best_W}

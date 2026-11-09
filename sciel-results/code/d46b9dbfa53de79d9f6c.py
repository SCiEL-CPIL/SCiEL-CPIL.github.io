"""Search for a rank-deficient tensor decomposition of 2x3x7 matrix multiplication.

Strategy:
  1. Start with the schoolbook rank-42 decomposition.
  2. Use a randomized greedy column-removal procedure: repeatedly try to remove
     one column (rank-1 term) from the factor matrices, checking whether the
     residual can still be exactly represented with the remaining columns.
  3. Also try a small amount of randomized local search on the factor matrices
     to find alternative representations that permit rank reduction.
  4. Return the best (lowest-rank) valid decomposition found within the time budget.
"""

import time
import random
import itertools

DIMENSIONS = [2, 3, 7]

def _build_schoolbook(n, m, p):
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

def _tensor_from_factors(U, V, W, n, m, p):
    """Build the full tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    Only valid if U, V, W have the same number of columns."""
    rank = len(U[0]) if U else 0
    T = [[0] * p for _ in range(n)]
    for j in range(m):
        T[j] = [[0] * p for _ in range(n)]
    # Actually T is 3D: T[i][j][k]
    T = [[[0] * p for _ in range(m)] for _ in range(n)]
    for c in range(rank):
        for i in range(n):
            u_val = U[i * m][c]
            if u_val == 0:
                continue
            for j in range(m):
                u_val2 = U[i * m + j][c]
                v_val = V[j * p][c]
                if u_val2 == 0 or v_val == 0:
                    continue
                for k in range(p):
                    v_val2 = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if v_val2 == 0 or w_val == 0:
                        continue
                    T[i][j][k] += u_val2 * v_val2 * w_val
    return T

def _tensor_from_factors_fast(U, V, W, n, m, p):
    """Faster: iterate over columns, multiply outer products."""
    rank = len(U[0]) if U else 0
    T = [[[0] * p for _ in range(m)] for _ in range(n)]
    for c in range(rank):
        u_col = [U[i * m + j][c] for i in range(n) for j in range(m)]
        v_col = [V[j * p + k][c] for j in range(m) for k in range(p)]
        w_col = [W[k * n + i][c] for k in range(p) for i in range(n)]
        for idx_u in range(n * m):
            u_val = u_col[idx_u]
            if u_val == 0:
                continue
            i = idx_u // m
            j = idx_u % m
            for idx_v in range(m * p):
                v_val = v_col[idx_v]
                if v_val == 0:
                    continue
                j2 = idx_v // p
                k = idx_v % p
                if j2 != j:
                    continue
                w_val = w_col[k * n + i]
                if w_val == 0:
                    continue
                T[i][j][k] += u_val * v_val * w_val
    return T

def _tensors_equal(T1, T2, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T1[i][j][k] - T2[i][j][k]) > tol:
                    return False
    return True

def _remove_column(U, V, W, c_to_remove):
    """Return U', V', W' with column c_to_remove removed."""
    U_new = [row[:c_to_remove] + row[c_to_remove+1:] for row in U]
    V_new = [row[:c_to_remove] + row[c_to_remove+1:] for row in V]
    W_new = [row[:c_to_remove] + row[c_to_remove+1:] for row in W]
    return U_new, V_new, W_new

def _check_rank_reducible(U, V, W, target_rank, n, m, p, deadline):
    """Try to find a subset of columns of size target_rank that still gives the same tensor.
    Uses greedy random removal."""
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return True, U, V, W
    T_target = _tensor_from_factors_fast(U, V, W, n, m, p)
    
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    
    columns = list(range(current_rank))
    random.shuffle(columns)
    
    removed = []
    for c in columns:
        if time.time() > deadline - 2:
            break
        if len(U_cur[0]) <= target_rank:
            break
        U_test, V_test, W_test = _remove_column(U_cur, V_cur, W_cur, c)
        T_test = _tensor_from_factors_fast(U_test, V_test, W_test, n, m, p)
        if _tensors_equal(T_test, T_target, n, m, p):
            U_cur, V_cur, W_cur = U_test, V_test, W_test
            removed.append(c)
    
    if len(U_cur[0]) <= target_rank:
        # Pad with zeros if needed (shouldn't be needed)
        return True, U_cur, V_cur, W_cur
    return False, None, None

def _random_perturbation(U, V, W, n, m, p, deadline):
    """Try random small perturbations to find alternative representations.
    This is a heuristic and may not help much, but we try a few."""
    rank = len(U[0])
    T_target = _tensor_from_factors_fast(U, V, W, n, m, p)
    best = (rank, U, V, W)
    
    for attempt in range(50):
        if time.time() > deadline - 2:
            break
        # Try to find a rank-1 dependency: can we write one column as a linear combo of others?
        # For now, just try removing random columns and checking
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Try to zero out one column and adjust others
        c = random.randint(0, rank - 1)
        # Check if column c is redundant
        U_rem, V_rem, W_rem = _remove_column(U, V, W, c)
        T_rem = _tensor_from_factors_fast(U_rem, V_rem, W_rem, n, m, p)
        if _tensors_equal(T_rem, T_target, n, m, p):
            best = (rank - 1, U_rem, V_rem, W_rem)
            return best
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # 20s safety margin
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try greedy column removal
    found, U2, V2, W2 = _check_rank_reducible(U, V, W, best_rank - 1, n, m, p, deadline)
    if found and U2 is not None:
        best_rank = len(U2[0])
        best_U, best_V, best_W = U2, V2, W2
    
    # Try a few more rounds of removal
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        found, U3, V3, W3 = _check_rank_reducible(best_U, best_V, best_W, best_rank - 1, n, m, p, deadline)
        if found and U3 is not None:
            best_rank = len(U3[0])
            best_U, best_V, best_W = U3, V3, W3
        else:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

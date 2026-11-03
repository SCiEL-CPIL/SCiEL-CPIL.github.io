"""Matrix multiplication 3x3 by 3x4 - rank minimization via combinatorial design search."""
import time
import math
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
DEADLINE = _START_TIME + 110  # 10s safety margin before 120s limit

def _time_remaining():
    return DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix using combinatorial design search.
    
    The decomposition requires: for all i,j,k
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook fallback (rank 36)
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Try ranks from 21 upward (combinatorial design suggests ~21-24 is achievable)
    ranks_to_try = list(range(21, 36)) + list(range(12, 21))
    
    for target_rank in ranks_to_try:
        if _time_remaining() < 3:
            break
        
        if best_rank <= target_rank:
            continue
        
        # More attempts for lower ranks (harder), fewer for higher (easier)
        max_attempts = 8 if target_rank >= 28 else (15 if target_rank >= 24 else 25)
        if _time_remaining() < 10:
            max_attempts = min(max_attempts, 3)
        
        for attempt in range(max_attempts):
            if _time_remaining() < 1.5:
                break
            
            try:
                result = _try_combinatorial_search(n, m, p, target_rank, attempt * 7919 + 104729)
                if result is not None:
                    best_result = result
                    best_rank = target_rank
                    break
            except Exception:
                continue
    
    return best_result

def _compute_M(U, V, n, m, p):
    """Compute M[i*p+k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]"""
    r = U.shape[1]
    M = np.zeros((n * p, r))
    for i in range(n):
        for k in range(p):
            for c in range(r):
                s = 0
                for j in range(m):
                    s += U[i * m + j, c] * V[j * p + k, c]
                M[i * p + k, c] = s
    return M

def _find_sparse_half_int_solution(M_row, r, max_support=2):
    """
    Find a sparse half-integer vector W such that M_row dot W = 1.
    Returns None if no such sparse solution is found.
    """
    # Single term: M_row[c] * W[c] = 1 => W[c] = 1/M_row[c]
    for c in range(r):
        m_val = M_row[c]
        if abs(m_val) > 1e-12:
            w_val = 1.0 / m_val
            if abs(w_val * 2 - round(w_val * 2)) < 1e-9:
                w = np.zeros(r)
                w[c] = round(w_val * 2) / 2.0
                return w
    
    if max_support < 2:
        return None
    
    # Two-term: M_row[a]*x + M_row[b]*y = 1 with x,y half-integers
    # Let X=2x, Y=2y (integers): M_row[a]*X + M_row[b]*Y = 2
    for a in range(r):
        ma = M_row[a]
        if abs(ma) < 1e-12:
            continue
        for b in range(a + 1, r):
            mb = M_row[b]
            if abs(mb) < 1e-12:
                continue
            
            # Try small integer X values
            for X in range(-8, 9):
                num = 2 - ma * X
                if abs(mb) > 1e-12:
                    Y = num / mb
                    if abs(Y - round(Y)) < 1e-9:
                        Y_int = int(round(Y))
                        if abs(X) <= 6 and abs(Y_int) <= 6:
                            w = np.zeros(r)
                            w[a] = X / 2.0
                            w[b] = Y_int / 2.0
                            return w
    
    return None

def _badness_count(M, n, p, r, max_support=2):
    """Count how many rows of M don't have a sparse half-integer solution."""
    bad = 0
    for idx in range(n * p):
        sol = _find_sparse_half_int_solution(M[idx], r, max_support)
        if sol is None:
            bad += 1
    return bad

def _try_combinatorial_search(n, m, p, target_rank, seed):
    """
    Combinatorial design search: find binary/sparse U, V patterns where
    the element-wise products M[i,k] = U_i ⊙ V_k have small support.
    """
    np.random.seed(seed)
    random.seed(seed)
    r = target_rank
    
    # Initialize U and V with random ±1 patterns
    # Ensure each column has at least 2 nonzeros in U and V
    U = np.random.choice([-1, 1], size=(n * m, r))
    V = np.random.choice([-1, 1], size=(m * p, r))
    
    # Ensure no column is all-zero after product (avoid trivial columns)
    for c in range(r):
        if np.all(U[:, c] == 0):
            U[np.random.randint(0, n * m), c] = 1
        if np.all(V[:, c] == 0):
            V[np.random.randint(0, m * p), c] = 1
    
    M = _compute_M(U, V, n, m, p)
    badness = _badness_count(M, n, p, r)
    
    if badness == 0:
        return _construct_W(M, U, V, n, m, p, r)
    
    # Local search: flip individual entries in U or V
    max_iters = 800
    patience = 0
    max_patience = 100
    
    for it in range(max_iters):
        if _time_remaining() < 1:
            break
        if badness == 0:
            break
        if patience > max_patience:
            break
        
        # Pick a random row and column to flip
        flip_U = np.random.rand() < 0.5
        if flip_U:
            idx = np.random.randint(0, n * m)
            col = np.random.randint(0, r)
            old_val = U[idx, col]
            U[idx, col] *= -1
        else:
            idx = np.random.randint(0, m * p)
            col = np.random.randint(0, r)
            old_val = V[idx, col]
            V[idx, col] *= -1
        
        new_M = _compute_M(U, V, n, m, p)
        new_badness = _badness_count(new_M, n, p, r)
        
        if new_badness < badness:
            badness = new_badness
            M = new_M
            patience = 0
        else:
            # Revert
            if flip_U:
                U[idx, col] = old_val
            else:
                V[idx, col] = old_val
            patience += 1
    
    if badness == 0:
        return _construct_W(M, U, V, n, m, p, r)
    
    return None

def _construct_W(M, U, V, n, m, p, r):
    """Construct W from M and verify the full decomposition."""
    W = np.zeros((p * n, r))
    for i in range(n):
        for k in range(p):
            sol = _find_sparse_half_int_solution(M[i * p + k], r)
            if sol is None:
                return None
            W[k * n + i] = sol
    
    # Convert to lists with exact half-integer values
    U_list = [[round(x * 2) / 2.0 for x in row] for row in U]
    V_list = [[round(x * 2) / 2.0 for x in row] for row in V]
    W_list = [[round(x * 2) / 2.0 for x in row] for row in W]
    
    # Verify exact decomposition
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > tol:
                    return False
    return True

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")

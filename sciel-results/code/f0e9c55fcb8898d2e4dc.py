import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    Uses exact Fraction arithmetic for safety.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _try_remove_term(U, V, W, n, m, p, term_idx):
    """
    Try to remove term term_idx by zeroing it out and checking if the rest still works.
    """
    rank = len(U[0])
    if rank <= 1:
        return None
    
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(len(U_new)):
        U_new[i][term_idx] = 0
    for i in range(len(V_new)):
        V_new[i][term_idx] = 0
    for i in range(len(W_new)):
        W_new[i][term_idx] = 0
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        U_compact = [row[:term_idx] + row[term_idx+1:] for row in U_new]
        V_compact = [row[:term_idx] + row[term_idx+1:] for row in V_new]
        W_compact = [row[:term_idx] + row[term_idx+1:] for row in W_new]
        return (U_compact, V_compact, W_compact)
    
    return None

def _try_pairwise_removal(U, V, W, n, m, p, idx1, idx2):
    """
    Try to remove two terms simultaneously.
    """
    rank = len(U[0])
    if rank <= 2:
        return None
    
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for i in range(len(U_new)):
        U_new[i][idx1] = 0
        U_new[i][idx2] = 0
    for i in range(len(V_new)):
        V_new[i][idx1] = 0
        V_new[i][idx2] = 0
    for i in range(len(W_new)):
        W_new[i][idx1] = 0
        W_new[i][idx2] = 0
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        cols_to_remove = sorted([idx1, idx2])
        U_compact = []
        V_compact = []
        W_compact = []
        for row in U_new:
            U_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        for row in V_new:
            V_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        for row in W_new:
            W_compact.append([row[j] for j in range(rank) if j not in cols_to_remove])
        return (U_compact, V_compact, W_compact)
    
    return None

def _construct_rank_28():
    """
    Constructs the known rank-28 decomposition for 2x3 * 3x5.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen (7 terms)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen (7 terms)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_find_lower_rank(U, V, W, n, m, p, deadline):
    """
    Attempts to find a lower rank decomposition by removing terms.
    Returns (U, V, W, rank) of the best found.
    """
    current_U, current_V, current_W = U, V, W
    current_rank = len(current_U[0])
    
    # Try single removals
    for term_idx in range(current_rank):
        if time.time() > deadline:
            break
        
        result = _try_remove_term(current_U, current_V, current_W, n, m, p, term_idx)
        if result is not None:
            current_U, current_V, current_W = result
            current_rank = len(current_U[0])
    
    # Try pairwise removals
    for pair in combinations(range(current_rank), 2):
        if time.time() > deadline:
            break
        
        result = _try_pairwise_removal(current_U, current_V, current_W, n, m, p, pair[0], pair[1])
        if result is not None:
            current_U, current_V, current_W = result
            current_rank = len(current_U[0])
    
    return current_U, current_V, current_W, current_rank

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 seconds, with safety margin
    
    n, m, p = 2, 3, 5
    
    # Construct the known rank-28 decomposition
    res_28 = _construct_rank_28()
    U = res_28['U']
    V = res_28['V']
    W = res_28['W']
    
    # Verify it to ensure correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Verification failed for rank-28 decomposition")
    
    # Try to find a lower rank decomposition
    U, V, W, rank = _try_find_lower_rank(U, V, W, n, m, p, deadline)
    
    return {'U': U, 'V': V, 'W': W}

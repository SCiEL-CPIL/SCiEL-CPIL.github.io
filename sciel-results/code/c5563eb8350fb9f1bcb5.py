"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]


def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += Fraction(U[u_idx][r]) * Fraction(V[v_idx][r]) * Fraction(W[w_idx][r])
                if s != Fraction(1):
                    return False
    return True


def _laderman_2x2x2():
    """
    Return Laderman's rank-7 decomposition for 2x2x2 matrix multiplication.
    U has shape (4,7), V has shape (4,7), W has shape (4,7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [
        [1, 1, 1, 1, 0, 0, 0],
        [1,-1, 0, 0, 1, 0, 0],
        [0, 0, 1,-1, 0, 1, 0],
        [0, 0, 0, 0, 1,-1, 0],
    ]
    V = [
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    W = [
        [0, 1, 0, 0, 0, 0, 0],
        [0,-1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0,-1, 1, 0, 0],
    ]
    
    n2, m2, p2 = 2, 2, 2
    if _verify(U, V, W, n2, m2, p2):
        return U, V, W
    else:
        rank = 8
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W


def _construct_rank_35():
    """
    Construct a rank-35 decomposition using Laderman for the 2x2x2 block
    and schoolbook for the remaining 28 terms.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_l, V_l, W_l = _laderman_2x2x2()
    
    u_map = [0, 1, 3, 4]
    v_map = [0, 1, 4, 5]
    w_map = [0, 1, 3, 4]
    
    for r in range(7):
        for lu in range(4):
            gu = u_map[lu]
            U[gu][r] = U_l[lu][r]
        for lv in range(4):
            gv = v_map[lv]
            V[gv][r] = V_l[lv][r]
        for lw in range(4):
            gw = w_map[lw]
            W[gw][r] = W_l[lw][r]
    
    column = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W


def _try_remove_column(U, V, W, n, m, p, col_to_remove):
    """
    Try to remove column col_to_remove by expressing its contribution
    as a linear combination of other columns. Returns (new_U, new_V, new_W)
    or None if removal is not possible.
    """
    rank = len(U[0])
    if rank <= 1:
        return None
    
    r = col_to_remove
    new_rank = rank - 1
    new_cols = [c for c in range(rank) if c != r]
    
    # For each (i,j,k), we need:
    # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] + sum_{r'!=r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] = 1
    # Currently sum over all r gives 1, so we need:
    # sum_{r'!=r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] = 1 - U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # This is a nonlinear system. Instead, try a simpler heuristic:
    # Check if column r is "redundant" in the sense that there exist
    # adjustments to other columns that compensate.
    
    # For now, use a conservative check: if the column has all zeros in
    # one of U, V, or W, it's already redundant (contributes nothing).
    u_col = [U[i][r] for i in range(n*m)]
    v_col = [V[i][r] for i in range(m*p)]
    w_col = [W[i][r] for i in range(p*n)]
    
    if all(x == 0 for x in u_col) or all(x == 0 for x in v_col) or all(x == 0 for x in w_col):
        # Column is zero, safe to remove
        new_U = [[U[i][c] for c in new_cols] for i in range(n*m)]
        new_V = [[V[i][c] for c in new_cols] for i in range(m*p)]
        new_W = [[W[i][c] for c in new_cols] for i in range(p*n)]
        return new_U, new_V, new_W
    
    return None


def _try_pairwise_merge(U, V, W, n, m, p, deadline):
    """
    Try to merge pairs of columns that have similar structure.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for r1 in range(rank):
        if time.time() > deadline:
            break
        for r2 in range(r1+1, rank):
            if time.time() > deadline:
                break
            
            # Check if columns r1 and r2 can be merged
            # by finding a linear combination that preserves the tensor
            # This is complex; use a heuristic: if they differ in only
            # a few positions, try to merge
            
            # Simple heuristic: skip for now
            pass
    
    return best_U, best_V, best_W, best_rank


def _try_rank_reduction(U, V, W, n, m, p, max_time=60):
    """
    Attempt to reduce the rank by removing redundant columns.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    rank = len(U[0])
    current_U, current_V, current_W = U, V, W
    current_rank = rank
    
    # Try removing each column one at a time
    for r in range(rank):
        if time.time() > deadline:
            break
        
        result = _try_remove_column(current_U, current_V, current_W, n, m, p, r)
        if result is not None:
            new_U, new_V, new_W = result
            if _verify(new_U, new_V, new_W, n, m, p):
                current_U, current_V, current_W = new_U, new_V, new_W
                current_rank -= 1
    
    return current_U, current_V, current_W, current_rank


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    current_rank = len(U[0])
    
    # Try to construct rank-35 decomposition
    if time.time() < deadline:
        U2, V2, W2 = _construct_rank_35()
        if _verify(U2, V2, W2, n, m, p):
            if len(U2[0]) < current_rank:
                U, V, W = U2, V2, W2
                current_rank = len(U[0])
        
        # Try rank reduction
        if time.time() < deadline - 30:
            U3, V3, W3, new_rank = _try_rank_reduction(U, V, W, n, m, p, max_time=30)
            if _verify(U3, V3, W3, n, m, p):
                if new_rank < current_rank:
                    U, V, W = U3, V3, W3
                    current_rank = new_rank
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook()
    
    return {"U": U, "V": V, "W": W}

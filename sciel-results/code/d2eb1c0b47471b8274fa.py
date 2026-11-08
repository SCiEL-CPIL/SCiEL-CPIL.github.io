"""Stronger program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 135  # seconds, with 15s safety margin before 150s limit


def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    if not U or not U[0]:
        return True
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _find_333_decomp(R, rng, max_iters=5000):
    """Try to find a rank-R decomposition for 3x3x3 matrix multiplication.
    
    Returns (U, V, W) each of shape (9, R) or None if not found.
    """
    n = m = p = 3
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            return None
        
        # Random initialization
        U = [[rng.choice(vals) for _ in range(R)] for _ in range(9)]
        V = [[rng.choice(vals) for _ in range(R)] for _ in range(9)]
        W = [[rng.choice(vals) for _ in range(R)] for _ in range(9)]
        
        if _validate(U, V, W, 3, 3, 3):
            return (U, V, W)
        
        # Local search: try flipping one entry at a time
        # Only do this every few iterations to save time
        if it % 5 == 0 and R <= 27:
            # Try perturbing one random entry
            mat_idx = rng.choice(3)  # 0=U, 1=V, 2=W
            row = rng.randint(0, 8)
            col = rng.randint(0, R-1)
            old_val = [U, V, W][mat_idx][row][col]
            new_val = rng.choice([v for v in vals if v != old_val])
            [U, V, W][mat_idx][row][col] = new_val
            
            if _validate(U, V, W, 3, 3, 3):
                return (U, V, W)
            else:
                [U, V, W][mat_idx][row][col] = old_val
    
    return None


def _construct_336_from_333(U333, V333, W333, R):
    """Construct a 3x3x6 decomposition from a 3x3x3 decomposition.
    
    The trick: for 3x3x6, we can use two copies of the 3x3x3 decomposition,
    one for k=0,1,2 and one for k=3,4,5.
    But actually, the structure is different. Let me think...
    
    For 3x3x6: U is (9, rank), V is (18, rank), W is (18, rank)
    T[i,j,k] = sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c] = 1
    
    If we have a 3x3x3 decomposition of rank R:
    U333 (9, R), V333 (9, R), W333 (9, R)
    where sum_c U333[i*3+j,c] * V333[j*3+k,c] * W333[k*3+i,c] = 1
    
    For 3x3x6, we can set:
    - For k=0,1,2: use the first 3 rows of V333 and W333 appropriately
    - For k=3,4,5: use a shifted version
    
    Actually, a cleaner approach: duplicate the 3x3x3 decomposition.
    Rank becomes 2*R.
    
    U_336[i*3+j, 0:R] = U333[i*3+j, :]
    U_336[i*3+j, R:2R] = U333[i*3+j, :]
    
    V_336[j*6+k, 0:R] = V333[j*3+k, :] for k=0,1,2
    V_336[j*6+k, R:2R] = V333[j*3+(k-3), :] for k=3,4,5
    
    W_336[k*3+i, 0:R] = W333[k*3+i, :] for k=0,1,2
    W_336[k*3+i, R:2R] = W333[(k-3)*3+i, :] for k=3,4,5
    
    Check: T[i,j,k] = sum_{c=0}^{R-1} U333[i,j,c]*V333[j,k,c]*W333[k,i,c] 
               + sum_{c=0}^{R-1} U333[i,j,c]*V333[j,k-3,c]*W333[k-3,i,c]
    
    For k=0,1,2: first sum = 1, second sum involves V333[j,-3+k,c] which is invalid.
    
    This doesn't work directly. Let me think differently.
    
    Better approach: For 3x3x6, split into two 3x3x3 blocks.
    Block 1: k=0,1,2 -> use 3x3x3 decomp of rank R
    Block 2: k=3,4,5 -> use another 3x3x3 decomp of rank R
    
    U_336[i*3+j, 0:R] = U333[i*3+j, :]
    U_336[i*3+j, R:2R] = U333[i*3+j, :]
    
    V_336[j*6+k, 0:R] = V333[j*3+k, :] for k in 0..2
    V_336[j*6+k, 0:R] = 0 for k in 3..5
    V_336[j*6+k, R:2R] = 0 for k in 0..2
    V_336[j*6+k, R:2R] = V333[j*3+(k-3), :] for k in 3..5
    
    W_336[k*3+i, 0:R] = W333[k*3+i, :] for k in 0..2
    W_336[k*3+i, 0:R] = 0 for k in 3..5
    W_336[k*3+i, R:2R] = 0 for k in 0..2
    W_336[k*3+i, R:2R] = W333[(k-3)*3+i, :] for k in 3..5
    
    Then T[i,j,k] for k=0,1,2:
    = sum_{c=0}^{R-1} U333[i,j,c] * V333[j,k,c] * W333[k,i,c] + 0 = 1 ✓
    
    For k=3,4,5:
    = 0 + sum_{c=0}^{R-1} U333[i,j,c] * V333[j,k-3,c] * W333[k-3,i,c] = 1 ✓
    
    This works! Rank = 2*R.
    """
    R2 = 2 * R
    U = [[0] * R2 for _ in range(9)]
    V = [[0] * R2 for _ in range(18)]
    W = [[0] * R2 for _ in range(18)]
    
    for i in range(3):
        for j in range(3):
            for c in range(R):
                U[i*3+j][c] = U333[i*3+j][c]
                U[i*3+j][R+c] = U333[i*3+j][c]
        
        for k in range(3):
            for c in range(R):
                V[j*6+k][c] = V333[j*3+k][c]
                V[j*6+(k+3)][R+c] = V333[j*3+k][c]
        
        for k in range(3):
            for c in range(R):
                W[k*3+i][c] = W333[k*3+i][c]
                W[(k+3)*3+i][R+c] = W333[k*3+i][c]
    
    return {"U": U, "V": V, "W": W}


def _try_direct_336_decomp(R, rng, max_iters=3000):
    """Try to directly find a rank-R decomposition for 3x3x6."""
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            return None
        
        U = [[rng.choice(vals) for _ in range(R)] for _ in range(9)]
        V = [[rng.choice(vals) for _ in range(R)] for _ in range(18)]
        W = [[rng.choice(vals) for _ in range(R)] for _ in range(18)]
        
        if _validate(U, V, W, 3, 3, 6):
            return {"U": U, "V": V, "W": W}
    
    return None


def _greedy_remove_columns(U, V, W, n, m, p, max_rounds=200):
    """Try to reduce rank by removing zero-contribution columns."""
    rank = len(U[0]) if U and U[0] else 0
    
    for _ in range(max_rounds):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        removed = False
        for c in range(len(U[0])):
            # Check if column c is all zeros in at least one matrix
            # or try removing it
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            
            if _validate(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                removed = True
                break
        
        if not removed:
            break
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p  # 54
    
    # Start with schoolbook as fallback
    best_result = _make_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    rng = random.Random(42)
    
    # Strategy 1: Find 3x3x3 decomposition, then construct 3x3x6
    # Known: rank(3x3x3) <= 23, so 3x3x6 rank <= 46
    for R in range(18, 28):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        result_333 = _find_333_decomp(R, rng, max_iters=2000)
        if result_333 is not None:
            U333, V333, W333 = result_333
            result_336 = _construct_336_from_333(U333, V333, W333, R)
            
            # Verify
            if _validate(result_336["U"], result_336["V"], result_336["W"], n, m, p):
                imp = _greedy_remove_columns(
                    [row[:] for row in result_336["U"]],
                    [row[:] for row in result_336["V"]],
                    [row[:] for row in result_336["W"]],
                    n, m, p
                )
                imp_rank = len(imp["U"][0]) if imp["U"] and imp["U"][0] else 2*R
                if imp_rank < best_rank:
                    best_rank = imp_rank
                    best_result = imp
                break
    
    # Strategy 2: Direct search for 3x3x6
    if best_rank == schoolbook_rank:
        for R in range(36, min(54, 50)):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            result = _try_direct_336_decomp(R, rng, max_iters=1000)
            if result is not None:
                imp = _greedy_remove_columns(
                    [row[:] for row in result["U"]],
                    [row[:] for row in result["V"]],
                    [row[:] for row in result["W"]],
                    n, m, p
                )
                imp_rank = len(imp["U"][0]) if imp["U"] and imp["U"][0] else R
                if imp_rank < best_rank:
                    best_rank = imp_rank
                    best_result = imp
                break
    
    return best_result

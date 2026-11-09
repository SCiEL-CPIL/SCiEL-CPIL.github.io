"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using integer arithmetic. Returns True if valid."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _try_eliminate_column(U, V, W, n, m, p, col_idx):
    """Try to eliminate a specific column by finding a rank-1 correction.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None)."""
    rank = len(U[0])
    if rank <= 1:
        return (False, None, None, None)
    
    # The contribution of column col_idx is:
    # For each (i,j,k): U[i*m+j, col_idx] * V[j*p+k, col_idx] * W[k*n+i, col_idx]
    # We want to find coefficients alpha, beta, gamma such that we can modify
    # other columns to cancel this contribution.
    
    # Simpler approach: check if the column is already redundant
    # (i.e., removing it still gives a valid decomposition)
    test_U = [row[:] for row in U]
    test_V = [row[:] for row in V]
    test_W = [row[:] for row in W]
    
    for i in range(len(test_U)):
        test_U[i].pop(col_idx)
    for i in range(len(test_V)):
        test_V[i].pop(col_idx)
    for i in range(len(test_W)):
        test_W[i].pop(col_idx)
    
    if _verify_fast(test_U, test_V, test_W, n, m, p):
        return (True, test_U, test_V, test_W)
    
    # Try to find a rank-1 update to other columns that cancels this column's contribution
    # This is complex, so we try a few random strategies
    for _ in range(10):
        # Pick a random other column to modify
        other_col = random.randint(0, rank - 1)
        if other_col == col_idx:
            continue
        
        # Try to adjust column other_col to cancel column col_idx's contribution
        # For each (i,j,k), we need:
        # U[i*m+j, other_col] * V[j*p+k, other_col] * W[k*n+i, other_col] 
        # to cancel U[i*m+j, col_idx] * V[j*p+k, col_idx] * W[k*n+i, col_idx]
        
        # This is a nonlinear system, so we try simple adjustments
        # Strategy: scale column other_col and add a multiple of col_idx
        # But this is getting complex. For now, just try removing.
        pass
    
    return (False, None, None, None)

def _greedy_rank_reduction(U, V, W, n, m, p, deadline):
    """Greedily try to eliminate columns."""
    current_rank = len(U[0])
    
    while current_rank > 1:
        if time.time() > deadline:
            break
        
        eliminated = False
        # Try each column
        for col_idx in range(current_rank - 1, -1, -1):
            if time.time() > deadline:
                break
            
            success, new_U, new_V, new_W = _try_eliminate_column(U, V, W, n, m, p, col_idx)
            if success:
                U = new_U
                V = new_V
                W = new_W
                current_rank -= 1
                eliminated = True
                break
        
        if not eliminated:
            break
    
    return U, V, W, current_rank

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 110  # Leave 10s margin
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try greedy rank reduction
    trial_U, trial_V, trial_W, trial_rank = _greedy_rank_reduction(
        [row[:] for row in U], [row[:] for row in V], [row[:] for row in W],
        n, m, p, deadline
    )
    
    if trial_rank < best_rank and _verify_fast(trial_U, trial_V, trial_W, n, m, p):
        best_rank = trial_rank
        best_U = trial_U
        best_V = trial_V
        best_W = trial_W
    
    # Final verification
    if _verify_fast(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback to schoolbook
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

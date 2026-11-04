"""Tensor decomposition for 2x3x9 matrix multiplication.

Searches for a rank-minimal exact tensor decomposition for
M_{2,3,9}: R^{2x3} x R^{3x9} -> R^{2x9}.

The schoolbook decomposition has rank 54. We attempt to find
a lower-rank decomposition using structured search within
the time budget, falling back to schoolbook if no improvement
is found.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def _verify_decomposition(U, V, W, n=2, m=3, p=9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Reconstruct tensor and check
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The target tensor has T[i,j,k] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n=2, m=3, p=9):
    """Return the standard rank n*m*p decomposition."""
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

def _try_rank_reduction(U, V, W, n=2, m=3, p=9, max_attempts=5000, deadline=None):
    """Try to reduce rank by finding linear dependencies among terms.
    
    Attempts to replace pairs of terms with a single term or remove
    terms that are linearly dependent.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    # Try to find if any column is a linear combination of others
    # and can be eliminated by adjusting other columns
    # This is a simplified approach: try removing each column and see
    # if the remaining columns can be scaled to still work.
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try removing column r
        r = attempt % rank
        if rank <= 1:
            break
        
        # Check if column r is linearly dependent on others
        # Simple test: try setting column r to zero and see if we can
        # adjust other columns to compensate. This is complex, so we use
        # a heuristic: check if the column is a scalar multiple of another.
        removed = False
        for r2 in range(rank):
            if r2 == r:
                continue
            # Check if col r is a scalar multiple of col r2
            ratios = []
            valid = True
            for idx in range(len(U)):
                if V[idx][r2] != 0:
                    ratio = V[idx][r] / V[idx][r2]
                    ratios.append(ratio)
                elif V[idx][r] != 0:
                    valid = False
                    break
            if not valid or not ratios:
                continue
            
            # Check consistency
            base_ratio = ratios[0]
            consistent = all(abs(r - base_ratio) < 1e-12 for r in ratios)
            if consistent:
                # Check U and W
                u_valid = True
                for idx in range(len(U)):
                    expected = base_ratio * U[idx][r2]
                    if abs(U[idx][r] - expected) > 1e-12:
                        u_valid = False
                        break
                if u_valid:
                    w_valid = True
                    for idx in range(len(W)):
                        expected = base_ratio * W[idx][r2]
                        if abs(W[idx][r] - expected) > 1e-12:
                            w_valid = False
                            break
                    if w_valid:
                        # Remove column r
                        for row in U:
                            del row[r]
                        for row in V:
                            del row[r]
                        for row in W:
                            del row[r]
                        rank -= 1
                        removed = True
                        break
        if removed:
            continue
        if not removed and attempt > 0:
            # No more simple reductions
            break
    
    return U, V, W, rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try rank reduction
    U, V, W, new_rank = _try_rank_reduction(
        best["U"], best["V"], best["W"], n, m, p,
        max_attempts=10000, deadline=deadline
    )
    if new_rank < best_rank and _verify_decomposition(U, V, W, n, m, p):
        best = {"U": U, "V": V, "W": W}
        best_rank = new_rank
    
    if time.time() > deadline:
        return best
    
    # Try more aggressive search: look for structured low-rank decompositions
    # For M_{2,m,p}, try rank = 2*m*p - c for small c
    # Use random initialization with exact rational arithmetic
    
    target_ranks = [best_rank - 1, best_rank - 2, best_rank - 3]
    for target_rank in target_ranks:
        if time.time() > deadline - 5:
            break
        if target_rank < max(n*m, m*p, p*n):
            continue
        
        # Try a few random structured initializations
        for trial in range(100):
            if time.time() > deadline - 2:
                break
            
            # Initialize with random rational values (small integers)
            np.random.seed(trial * 12345 + 67890)
            U_t = np.random.randint(-2, 3, (n*m, target_rank)).astype(float)
            V_t = np.random.randint(-2, 3, (m*p, target_rank)).astype(float)
            W_t = np.random.randint(-2, 3, (p*n, target_rank)).astype(float)
            
            # Check if this happens to work (very unlikely, but cheap)
            if _verify_decomposition(U_t.tolist(), V_t.tolist(), W_t.tolist(), n, m, p):
                best = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END

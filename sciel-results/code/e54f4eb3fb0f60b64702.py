"""Matrix multiplication 3x5x5: find rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product
from fractions import Fraction
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            expected = 1 if (i==ip and j==jp and k==kp) else 0
                            s = 0
                            for c in range(r):
                                s += U[i*m+j, c] * V[jp*p+kp, c] * W[kp*n+ip, c]
                            if s != expected:
                                return False
    return True

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen. U:4x7, V:4x7, W:4x7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
    # M3
    U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    # M4
    U[3][3]=1; V[1][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # M5
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # M6
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _build_strassen_blocks(n, m, p, block_list):
    """Build decomposition using given 2x2x2 Strassen blocks + schoolbook for rest."""
    in_block = set()
    for (i0, j0, k0) in block_list:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0+di < n and j0+dj < m and k0+dk < p:
                        in_block.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(in_block)
    n_strassen = 7 * len(block_list)
    new_rank = n_school + n_strassen
    new_U = [[0]*new_rank for _ in range(n*m)]
    new_V = [[0]*new_rank for _ in range(m*p)]
    new_W = [[0]*new_rank for _ in range(p*n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in in_block:
                    new_U[i*m+j][col_idx] = 1
                    new_V[j*p+k][col_idx] = 1
                    new_W[k*n+i][col_idx] = 1
                    col_idx += 1
    
    sU, sV, sW = _strassen_2x2x2()
    for (i0, j0, k0) in block_list:
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    val = sU[li*2+lj][c]
                    if val != 0:
                        gi, gj = i0+li, j0+lj
                        new_U[gi*m+gj][col_idx] += val
            for lj in range(2):
                for lk in range(2):
                    val = sV[lj*2+lk][c]
                    if val != 0:
                        gj, gk = j0+lj, k0+lk
                        new_V[gj*p+gk][col_idx] += val
            for lk in range(2):
                for li in range(2):
                    val = sW[lk*2+li][c]
                    if val != 0:
                        gk, gi = k0+lk, i0+li
                        new_W[gk*n+gi][col_idx] += val
            col_idx += 1
    
    return new_U, new_V, new_W

def _try_overlapping_strassen(n, m, p):
    """Try overlapping Strassen blocks in i-dimension to get better rank."""
    # For 3x5x5: use overlapping 2x2x2 blocks
    # Block A: i=0,1; j=0,1; k=0,1
    # Block B: i=1,2; j=0,1; k=0,1  (overlaps in i=1)
    # But overlapping doesn't directly work since Strassen is for disjoint indices.
    # Instead, try: treat as two 2x5x5 problems sharing structure.
    # 2x5x5 = two 2x2x2 blocks + one 2x1x5 strip
    # Rank of 2x5x5 via Strassen: 2*7 + 2*5 = 24 (vs 50 schoolbook)
    # 3x5x5: use i=0,1 as 2x5x5 (rank 24) + i=2 row (rank 25) = 49?
    # But that's not quite right. Let me think differently.
    
    # Better structural approach: use the fact that we can apply Strassen
    # to overlapping 2x2 blocks in a way that the shared index contributes
    # to both. This is the "Winograd" style approach.
    
    # For now, try: 4 disjoint blocks in j,k dims for i=0,1
    # Plus handle i=2 separately
    # i=0,1: j=0-4, k=0-4. Use 2x2 blocks at (j=0,1),(j=2,3) x (k=0,1),(k=2,3)
    # That's 4 blocks of 2x2x2 for i=0,1. Each saves 1. Total save 4.
    # Remaining: i=0,1 j=4 k=0-4 (10 terms) + i=0,1 j=0-3 k=4 (8 terms) + i=2 all (25 terms)
    # Total: 4*7 + 10 + 8 + 25 = 28+43 = 71. Same as before.
    
    # Try different: use 2x2x2 blocks but also try to combine the "leftover" strips
    # with Strassen-like cancellations.
    
    # Actually, let me try a completely different approach:
    # Use the known lower bound. For 3x5x5, the rank is at least... 
    # Let me just try to find if we can do better than 71 with a randomized search.
    
    return None

def _randomized_search(n, m, p, target_rank, n_trials=100, timeout=60):
    """Try to find a decomposition of given rank via random initialization + projection."""
    start = time.time()
    # This is a heuristic: initialize with schoolbook and try to zero out columns
    # by finding linear dependencies.
    
    # Start from schoolbook
    U = np.zeros((n*m, n*m*p), dtype=object)
    V = np.zeros((m*p, n*m*p), dtype=object)
    W = np.zeros((p*n, n*m*p), dtype=object)
    for idx, (i,j,k) in enumerate(product(range(n), range(m), range(p))):
        U[i*m+j, idx] = 1
        V[j*p+k, idx] = 1
        W[k*n+i, idx] = 1
    
    # Try to find linear dependencies among the rank-1 tensors
    # This is complex; for now, return schoolbook
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline
    rank_school = n * m * p  # 75
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Strategy 1: 4 disjoint Strassen 2x2x2 blocks (rank 71)
    blocks1 = [(0,0,0), (0,2,0), (0,0,2), (0,2,2)]
    U1, V1, W1 = _build_strassen_blocks(n, m, p, blocks1)
    if _verify(U1, V1, W1, n, m, p):
        best = {"U": U1, "V": V1, "W": W1}
        best_rank = 71
    
    if time.time() - start_time > 120:
        return best
    
    # Strategy 2: Try 5 Strassen blocks by using overlapping in a valid way
    # For 3x5x5: blocks at (0,0,0),(0,2,0),(0,0,2),(0,2,2) for i=0,1
    # Can we add a block for i=1,2? Block at (1,0,0) uses i=1,2; j=0,1; k=0,1
    # This overlaps with block (0,0,0) at i=1,j=0,1,k=0,1
    # Overlapping blocks don't simply add up. Need a different approach.
    
    # Strategy 3: Use a 3x3x3 sub-problem approach
    # 3x3x3 has known rank 23 (Bini et al.)
    # 3x5x5 = 3x3x3 (rank 23) + strips
    # Strips: 3x2x5 (i all, j=3,4, k all) = 30 terms
    #         3x3x2 (i all, j=0,1,2, k=3,4) = 18 terms
    # Total: 23 + 30 + 18 = 71. Same.
    # But maybe the strips can be optimized too.
    
    # Strategy 4: Try to find a rank-70 decomposition by modifying the Strassen approach
    # to allow one additional cancellation across blocks.
    
    # For now, return best found (rank 71)
    return best
# EVOLVE-BLOCK-END

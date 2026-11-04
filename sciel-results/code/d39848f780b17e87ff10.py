"""Matrix multiplication 4x5x5 - structured rank reduction search."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np
from itertools import product, combinations

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 140  # Safety margin
    
    # Build schoolbook as baseline
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    
    # Verify schoolbook
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Verify schoolbook is valid
    assert verify(U_sb, V_sb, W_sb, schoolbook_rank), "Schoolbook verification failed"
    
    # Strategy: Use structured rank reduction via bilinear algebra
    # For matrix multiplication, we can exploit the fact that the tensor
    # has a specific structure. We'll try to find a lower-rank decomposition
    # by looking for linear dependencies among the schoolbook terms.
    
    # Key approach: Try to merge pairs of terms that share indices.
    # For terms (i,j,k) and (i,j',k') where i is shared, we can potentially
    # combine them if we can find coefficients that satisfy all constraints.
    
    # A more systematic approach: use the fact that for any fixed i,
    # the sub-tensor for that i has rank at most m*p = 25.
    # We can try to find a decomposition that exploits this structure.
    
    # Given the difficulty of finding rank < 100 for 4x5x5, let's try
    # a targeted search for small rank reductions using algebraic methods.
    
    # Approach: For each pair of schoolbook terms that share at least one index,
    # check if we can merge them into a single term with adjusted coefficients.
    
    # Let's try a different strategy: use the "border rank" concept.
    # The border rank of the matrix multiplication tensor might be lower
    # than the tensor rank, and we might find a valid decomposition
    # with rank slightly below 100.
    
    # For now, let's try a systematic search for rank 99.
    # We'll look for a single term that can be eliminated by adjusting
    # other terms in a way that maintains all constraints.
    
    import random
    random.seed(42)
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a valid rank-99 decomposition
    # We'll try each possible term to eliminate and see if we can fix it
    for col_to_remove in range(schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # Determine which (i,j,k) this column corresponds to
        i0 = col_to_remove // (m*p)
        j0 = (col_to_remove // p) % m
        k0 = col_to_remove % p
        
        # Create a copy without this column
        U = [row[:] for row in U_sb]
        V = [row[:] for row in V_sb]
        W = [row[:] for row in W_sb]
        
        # We need to find adjustments to other columns such that
        # the constraint at (i0,j0,k0) is still satisfied.
        # The missing contribution is 1 at (i0,j0,k0).
        
        # Try to distribute this 1 among other terms that share
        # at least one index with (i0,j0,k0).
        
        # For each other term (i,j,k) that shares an index, we can
        # potentially adjust its coefficient.
        
        # Let's try a simple approach: find two terms that share
        # indices and set their coefficients to 0.5 each.
        
        # Find candidate terms
        candidates = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i,j,k) == (i0,j0,k0):
                        continue
                    shares = 0
                    if i == i0: shares += 1
                    if j == j0: shares += 1
                    if k == k0: shares += 1
                    if shares > 0:
                        col_idx = i*m*p + j*p + k
                        candidates.append((i,j,k,col_idx))
        
        if len(candidates) >= 2:
            # Try all pairs of candidates (limited for time)
            for c1 in candidates[:10]:
                if time.time() > deadline - 3:
                    break
                for c2 in candidates[:10]:
                    if c1[3] == c2[3]:
                        continue
                    
                    # Try setting both to 0.5
                    U2 = [row[:] for row in U]
                    V2 = [row[:] for row in V]
                    W2 = [row[:] for row in W]
                    
                    # Set the removed column to 0
                    for r in range(nm):
                        U2[r][col_to_remove] = 0
                    for r in range(mp):
                        V2[r][col_to_remove] = 0
                    for r in range(pn):
                        W2[r][col_to_remove] = 0
                    
                    # Set the two candidate columns to 0.5
                    col1, col2 = c1[3], c2[3]
                    U2[c1[0]*m + c1[1]][col1] = 0.5
                    U2[c2[0]*m + c2[1]][col2] = 0.5
                    V2[c1[1]*p + c1[2]][col1] = 0.5
                    V2[c2[1]*p + c2[2]][col2] = 0.5
                    W2[c1[2]*n + c1[0]][col1] = 0.5
                    W2[c2[2]*n + c2[0]][col2] = 0.5
                    
                    # Check if this is valid
                    if verify(U2, V2, W2, schoolbook_rank):
                        # Remove the zero column
                        U3 = [[U2[r][c] for c in range(schoolbook_rank) if c != col_to_remove] for r in range(nm)]
                        V3 = [[V2[r][c] for c in range(schoolbook_rank) if c != col_to_remove] for r in range(mp)]
                        W3 = [[W2[r][c] for c in range(schoolbook_rank) if c != col_to_remove] for r in range(pn)]
                        
                        if verify(U3, V3, W3, schoolbook_rank - 1):
                            best_rank = schoolbook_rank - 1
                            best_U, best_V, best_W = U3, V3, W3
                            break
                if best_rank < schoolbook_rank:
                    break
        if best_rank < schoolbook_rank:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

"""Tensor decomposition for 2x3 x 3x3 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    # The target tensor: T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try rank 17 first
    best_result = None
    best_rank = n * m * p  # 18 schoolbook
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Schoolbook as fallback
    fallback = schoolbook()
    best_rank = n * m * p
    
    # Attempt to find rank 17 decomposition
    # Use a structured approach: try Winograd-like transformations
    
    # For the j-dimension (size 3), use the transform:
    # Split into pairs and use (a+b), (a-b), c pattern
    # This gives 4 multiplies per (i,k) pair instead of 3
    # Total: 2 * 3 * 4 = 24... that's worse.
    
    # Alternative: try to find rank 17 by optimization
    # Set up the 18 equations and solve for 17 rank-1 terms
    
    # Let's try a specific construction based on known results
    # The rank of M_{2,3,3} is 17.
    
    # Try: use 17 terms where one term covers multiple (i,j,k) points
    # via clever coefficient choices
    
    # Direct approach: parametrize 17 terms and solve
    # This is a system of 18 polynomial equations in 17*15 = 255 variables
    # Too large for direct symbolic, but we can try numerical + rounding
    
    # Try a structured rank-17 ansatz:
    # 16 terms are "standard" for all (i,j,k) except (0,0,0)
    # 1 term is a correction that fixes (0,0,0) and doesn't break others
    
    # Actually, let me try the following known construction:
    # Use the fact that we can write one product as a sum of others
    
    # Let me try: for the j=0,1,2 dimension, use:
    # A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # = (A[i,0]+A[i,1])*B[0,k] + A[i,0]*(B[1,k]-B[0,k]) + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # That's still 4 terms, not helpful.
    
    # Let me try a different split:
    # = (A[i,0]+A[i,2])*B[0,k] + A[i,0]*(B[1,k]-B[0,k]) + A[i,1]*B[1,k] + A[i,2]*(B[2,k]-B[0,k])
    # Still 4.
    
    # The key insight for rank 17: one of the 18 "cells" can be shared.
    
    # Try: modify the (0,0,0) term to also contribute to (0,1,0) or similar
    # by making V[0,0,c] and V[1,0,c] both nonzero.
    
    # Let's try a specific ansatz with 17 terms:
    # Terms 1-16: standard for all (i,j,k) except (0,0,0) and (0,0,1)
    # Term 17: a combined term that handles both (0,0,0) and (0,0,1)
    
    # For term 17 to handle (0,0,0) and (0,0,1):
    # U[0, c17] * V[0,0, c17] * W[0,0, c17] = 1  (for (0,0,0))
    # U[0, c17] * V[0,1, c17] * W[1,0, c17] = 1  (for (0,0,1))
    # And for all other (i',j',k') != (0,0,0),(0,0,1): contribution = 0
    
    # This requires U[0,c17] != 0, and:
    # V[0,0,c17]*W[0,0,c17] = 1/U[0,c17]
    # V[0,1,c17]*W[1,0,c17] = 1/U[0,c17]
    # V[j',k',c17]*W[k',i',c17] = 0 for all other (i',j',k')
    
    # For (i'=0, j'=0, k'=0): V[0,0,c17]*W[0,0,c17] = 1/U[0,c17] ✓
    # For (i'=0, j'=0, k'=1): V[0,1,c17]*W[1,0,c17] = 1/U[0,c17] ✓
    # For (i'=0, j'=0, k'=2): V[0,2,c17]*W[2,0,c17] = 0
    # For (i'=0, j'=1, k'=0): V[1,0,c17]*W[0,0,c17] = 0
    # For (i'=0, j'=1, k'=1): V[1,1,c17]*W[1,0,c17] = 0
    # etc.
    
    # From V[0,2,c17]*W[2,0,c17] = 0 and V[1,0,c17]*W[0,0,c17] = 0:
    # Since W[0,0,c17] != 0 (from first eq), V[1,0,c17] = 0
    # Similarly, V[1,1,c17] = 0 (from V[1,1,c17]*W[1,0,c17]=0, W[1,0,c17]!=0)
    # V[2,0,c17] = 0, V[2,1,c17] = 0
    # From V[0,2,c17]*W[2,0,c17] = 0: either V[0,2,c17]=0 or W[2,0,c17]=0
    # From (i'=1, j'=0, k'=0): V[0,0,c17]*W[0,1,c17] = 0, so W[0,1,c17]=0
    # From (i'=1, j'=0, k'=1): V[0,1,c17]*W[1,1,c17] = 0, so W[1,1,c17]=0
    # From (i'=1, j'=0, k'=2): V[0,2,c17]*W[2,1,c17] = 0
    # From (i'=1, j'=1, k'=0): V[1,0,c17]*W[0,1,c17] = 0 (already have V[1,0]=0)
    
    # So W is nonzero only at W[0,0,c17] and W[1,0,c17]
    # V is nonzero only at V[0,0,c17], V[0,1,c17], possibly V[0,2,c17] if W[2,0,c17]=0
    
    # Let V[0,2,c17] = 0 for simplicity.
    # Then: V[0,0,c17]*W[0,0,c17] = 1/U[0,c17]
    #        V[0,1,c17]*W[1,0,c17] = 1/U[0,c17]
    
    # Now check: does this break any other equation?
    # For (i'=0, j'=0, k'=0): only term 17 contributes (since we removed standard terms for (0,0,0) and (0,0,1))
    #   Contribution: U[0,c17]*V[0,0,c17]*W[0,0,c17] = 1 ✓
    # For (i'=0, j'=0, k'=1): 
    #   Contribution: U[0,c17]*V[0,1,c17]*W[1,0,c17] = 1 ✓
    # For (i'=0, j'=1, k'=0): 
    #   Standard term (0,1,0) contributes: 1
    #   Term 17: U[0,c17]*V[1,0,c17]*W[0,0,c17] = U[0,c17]*0*W[0,0,c17] = 0 ✓
    # For (i'=0, j'=1, k'=1):
    #   Standard term (0,1,1) contributes: 1
    #   Term 17: U[0,c17]*V[1,1,c17]*W[1,0,c17] = 0 ✓
    
    # This works! Let me set:
    # U[0,c17] = 1
    # V[0,0,c17] = 1, W[0,0,c17] = 1
    # V[0,1,c17] = 1, W[1,0,c17] = 1
    # All other entries of U,V,W for c17 are 0
    
    # But wait: for (i'=0, j'=0, k'=1), the standard term would have contributed 1.
    # We removed it, so term 17 must provide 1. ✓
    # For (i'=0, j'=0, k'=0), standard removed, term 17 provides 1. ✓
    
    # Now check all other (i',j',k') are still 1 from their standard terms.
    # The standard terms cover all 18 points except (0,0,0) and (0,0,1).
    # Term 17 adds 1 at (0,0,0) and (0,0,1), and 0 elsewhere.
    
    # Total rank: 16 + 1 = 17!
    
    # Let me verify this more carefully.
    # Standard terms: for each (i,j,k) in {(0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),
    #   (0,2,0),(0,2,1),(0,2,2),(1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),(1,1,2),(1,2,0),(1,2,1),(1,2,2)}
    #   except (0,0,0) and (0,0,1), so 16 terms.
    # Term 17: handles (0,0,0) and (0,0,1).
    
    # Verify (0,0,0): sum_c U[0,c]*V[0,0,c]*W[0,0,c]
    #   Standard terms: none for (0,0,0) since we removed it
    #   Term 17: U[0,16]*V[0,0,16]*W[0,0,16] = 1*1*1 = 1 ✓
    
    # Verify (0,0,1): sum_c U[0,c]*V[0,1,c]*W[1,0,c]
    #   Standard terms: none for (0,0,1) since we removed it
    #   Term 17: U[0,16]*V[0,1,16]*W[1,0,16] = 1*1*1 = 1 ✓
    
    # Verify (0,0,2): sum_c U[0,c]*V[0,2,c]*W[2,0,c]
    #   Standard term (0,0,2): 1*1*1 = 1
    #   Term 17: U[0,16]*V[0,2,16]*W[2,0,16] = 1*0*0 = 0
    #   Total: 1 ✓
    
    # Verify (0,1,0): sum_c U[0,c]*V[1,0,c]*W[0,0,c]
    #   Standard term (0,1,0): 1*1*1 = 1
    #   Term 17: U[0,16]*V[1,0,16]*W[0,0,16] = 1*0*1 = 0
    #   Total: 1 ✓
    
    # Verify (1,0,0): sum_c U[1,c]*V[0,0,c]*W[0,1,c]
    #   Standard term (1,0,0): 1*1*1 = 1
    #   Term 17: U[1,16]*V[0,0,16]*W[0,1,16] = 0*1*0 = 0 (U[1,16]=0, W[0,1,16]=0)
    #   Total: 1 ✓
    
    # Verify (1,0,1): sum_c U[1,c]*V[0,1,c]*W[1,1,c]
    #   Standard term (1,0,1): 1
    #   Term 17: U[1,16]*V[0,1,16]*W[1,1,16] = 0*1*0 = 0
    #   Total: 1 ✓
    
    # All checks pass!
    
    rank = 17
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 0 and j == 0 and k in (0, 1):
                    continue  # skip (0,0,0) and (0,0,1)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # col should be 16 now
    # Add the combined term at col=16
    # U[0, 16] = 1 (i=0, j=0 → row 0)
    U[0][16] = 1
    # V[0,0, 16] = 1 (j=0, k=0 → row 0)
    V[0][16] = 1
    # V[0,1, 16] = 1 (j=0, k=1 → row 1)
    V[1][16] = 1
    # W[0,0, 16] = 1 (k=0, i=0 → row 0)
    W[0][16] = 1
    # W[1,0, 16] = 1 (k=1, i=0 → row 1)
    W[1][16] = 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    if max_err < 1e-10:
        return result
    else:
        # Fallback to schoolbook
        return fallback

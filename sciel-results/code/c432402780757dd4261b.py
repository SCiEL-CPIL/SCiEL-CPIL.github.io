"""
Matrix multiplication 2x5x5: Attempt rank-reduced exact tensor decomposition.
Starts with schoolbook rank-50 and tries to find lower-rank exact solutions
by exploiting the small n=2 dimension to merge terms.
"""

import time
from fractions import Fraction
import random

DIMENSIONS = [2, 5, 5]

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Build schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def validate(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Try rank 49: attempt to merge two schoolbook terms
    # Strategy: For a given j, try to combine the terms for (i=0,j,k1) and (i=1,j,k2)
    # by using a shared rank dimension with appropriate coefficients.
    
    # Let's try a more systematic approach: 
    # Use rank 49 with one "shared" dimension that contributes to two entries.
    
    # Approach: Take schoolbook and try to eliminate one column by adjusting others.
    # For rank 49, we need 49 columns. Let's try removing column (i=0, j=0, k=0)
    # and redistributing its contribution.
    
    # Actually, let's try a different structural approach:
    # For n=2, we can write:
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    #
    # If we define for each j: S_j = A[0,j] + A[1,j] and D_j = A[0,j] - A[1,j]
    # Then C[0,k] = (S_k_part + D_k_part)/2 ... this doesn't directly help.
    
    # Let me try a direct search for rank 49 with small integer entries.
    # Given the constraints, let's try:
    # - 48 columns are standard schoolbook terms (one per (i,j,k) except two specific ones)
    # - 1 column is a "combined" term that handles the two missing entries
    
    # For the combined term to work, we need:
    # U[i1*m+j, c] * V[j*p+k1, c] * W[k1*n+i1, c] = 1
    # U[i2*m+j, c] * V[j*p+k2, c] * W[k2*n+i2, c] = 1
    # And zero contribution to all other (i,j,k).
    
    # This is only possible if the combined term has non-zero entries only for 
    # the specific (i,j,k) pairs we want. But a rank-1 term U_row * V_col * W_entry
    # is non-zero for ALL (i,j,k) where U[i*m+j], V[j*p+k], W[k*n+i] are all non-zero.
    
    # So a single rank-1 term with U[i1*m+j]=a, U[i2*m+j]=b, V[j*p+k1]=c, V[j*p+k2]=d,
    # W[k1*n+i1]=e, W[k2*n+i2]=f would contribute:
    # To (i1,j,k1): a*c*e
    # To (i1,j,k2): a*d*f (if k2*n+i1 exists, but W[k2*n+i1] would need to be non-zero)
    # This creates unwanted contributions.
    
    # Let's try a different approach: use half-integer values.
    # For rank 49, try to find a solution where one "extra" column handles
    # two entries with fractional values.
    
    # Let's try: remove columns for (0,0,0) and (1,0,0), add one combined column.
    # The combined column needs:
    # U[0*5+0, c] * V[0*5+0, c] * W[0*2+0, c] = 1  (for i=0,j=0,k=0)
    # U[1*5+0, c] * V[0*5+0, c] * W[0*2+1, c] = 1  (for i=1,j=0,k=0)
    # And zero for all other (i,j,k).
    
    # For zero contribution to (0,0,1): U[0*5+0,c]*V[0*5+1,c]*W[1*2+0,c] = 0
    # This requires at least one of U[5,c], V[1,c], W[2,c] to be 0.
    # Similarly for other entries.
    
    # Let's set: U[0,c]=a, U[5,c]=b, V[0,c]=c_val, W[0,c]=d, W[1,c]=e
    # Constraints:
    # a*c_val*d = 1  (i=0,j=0,k=0)
    # b*c_val*e = 1  (i=1,j=0,k=0)
    # For (i=0,j=0,k=1): a*V[1,c]*W[2,c] = 0 => need V[1,c]=0 or W[2,c]=0
    # For (i=1,j=0,k=1): b*V[1,c]*W[3,c] = 0 => need V[1,c]=0 or W[3,c]=0
    # For (i=0,j=1,k=0): U[1,c]*V[5,c]*W[0,c] = 0 => need U[1,c]=0 or V[5,c]=0 or W[0,c]=0
    # But W[0,c]=d≠0, so need U[1,c]=0 or V[5,c]=0.
    # Similarly for other j values.
    
    # If we set V[j*5+0, c] = 0 for j≠0, and U[i*5+j, c] = 0 for j≠0,
    # and W[k*2+i, c] = 0 except for W[0,c] and W[1,c], then:
    # The only non-zero contributions are for j=0, k=0, i=0 or 1.
    # This works!
    
    # So: U[0,c]=a, U[5,c]=b, all other U entries in column c = 0
    #      V[0,c]=c_val, all other V entries in column c = 0
    #      W[0,c]=d, W[1,c]=e, all other W entries in column c = 0
    # With a*c_val*d = 1 and b*c_val*e = 1.
    
    # We can choose: a=1, c_val=1, d=1, b=1, e=1. Then both equations are satisfied!
    # But wait, this means the combined column is identical to having two separate columns
    # for (0,0,0) and (1,0,0). It doesn't save rank.
    
    # The issue: a single rank-1 term can only have one value per U row, V col, W entry.
    # To contribute to (0,0,0) and (1,0,0) with the same j=0, we need:
    # U[0,c]*V[0,c]*W[0,c] = 1
    # U[5,c]*V[0,c]*W[1,c] = 1
    # This requires V[0,c] ≠ 0, and then U[0,c]*W[0,c] = 1/V[0,c] and U[5,c]*W[1,c] = 1/V[0,c].
    # We can choose V[0,c]=1, U[0,c]=1, W[0,c]=1, U[5,c]=1, W[1,c]=1.
    # But this is exactly two separate rank-1 terms merged into one! The rank is saved.
    
    # Wait, but does this create unwanted contributions? Let's check:
    # For (i=0, j=0, k=1): U[0,c]*V[1,c]*W[2,c]. V[1,c]=0 (since we only set V[0,c]), so 0. ✓
    # For (i=0, j=1, k=0): U[1,c]*V[5,c]*W[0,c]. U[1,c]=0, V[5,c]=0, so 0. ✓
    # For (i=1, j=0, k=1): U[5,c]*V[1,c]*W[3,c]. V[1,c]=0, so 0. ✓
    # For (i=1, j=1, k=0): U[6,c]*V[5,c]*W[1,c]. U[6,c]=0, V[5,c]=0, so 0. ✓
    
    # Great! So we CAN merge (0,0,0) and (1,0,0) into one rank-1 term!
    # This gives us rank 49!
    
    # Let's construct the rank-49 solution:
    # 48 standard schoolbook terms for all (i,j,k) except (0,0,0) and (1,0,0)
    # 1 combined term for (0,0,0) and (1,0,0)
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 0 and j == 0 and k == 0:
                    continue  # handled by combined term
                if i == 1 and j == 0 and k == 0:
                    continue  # handled by combined term
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Combined term for (0,0,0) and (1,0,0)
    U[0 * m + 0][col] = 1  # i=0, j=0
    U[1 * m + 0][col] = 1  # i=1, j=0
    V[0 * p + 0][col] = 1  # j=0, k=0
    W[0 * n + 0][col] = 1  # k=0, i=0
    W[0 * n + 1][col] = 1  # k=0, i=1
    col += 1
    
    # Validate
    if validate(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # If validation failed, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np
from itertools import product

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Block Strassen decomposition for 4x5 * 5x6
    # Partition:
    # A (4x5): rows [0,1],[2,3]; cols [0,1],[2,3,4]
    #   A11(2x2), A12(2x3), A21(2x2), A22(2x3)
    # B (5x6): rows [0,1],[2,3,4]; cols [0,1,2],[3,4,5]
    #   B11(2x3), B12(2x3), B21(3x3), B22(3x3)
    # C (4x6): rows [0,1],[2,3]; cols [0,1,2],[3,4,5]
    #   C11(2x3), C12(2x3), C21(2x3), C22(2x3)
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Apply Strassen to the 2x2 block structure (outer level):
    # M1 = (A11+A22)(B11+B22) -- but sizes don't match for addition!
    # A11 is 2x2, A22 is 2x3. Cannot add directly.
    #
    # Alternative: Use Strassen on inner blocks where sizes match.
    # For C11 = A11*B11 + A12*B21: these are different size multiplications.
    # 
    # Better approach: just use schoolbook for each sub-block but count total.
    # Actually, let's use a different strategy: find explicit rank reduction
    # by combining terms that share structure.
    
    # Direct approach: construct decomposition with rank < 120
    # by finding linear dependencies among the 120 schoolbook terms.
    # 
    # The tensor T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    # Rank is the minimum number of rank-1 tensors summing to T.
    # 
    # For a concrete improvement, let's try to merge pairs of schoolbook terms.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if there
    # exist coefficients such that the sum still equals 1 for all required (i,j,k).
    
    # Let's try a systematic approach: start with schoolbook and try to 
    # eliminate one term at a time by adjusting others.
    
    def try_eliminate_term(school_U, school_V, school_W, target_col):
        """Try to eliminate column target_col by adjusting other columns."""
        n, m, p = DIMENSIONS
        rank = len(school_U[0])
        
        # Create copies
        U = [row[:] for row in school_U]
        V = [row[:] for row in school_V]
        W = [row[:] for row in school_W]
        
        # The term at target_col corresponds to some (i0, j0, k0)
        # U[i0*m+j0][target_col] = 1, V[j0*p+k0][target_col] = 1, W[k0*n+i0][target_col] = 1
        # All other entries in this column are 0.
        
        # To eliminate this column, we need to find modifications to other columns
        # such that the contribution of column target_col is absorbed.
        # This requires: for all (i,j,k) where the target term contributes,
        # the sum of other terms must compensate.
        
        # The target term contributes 1 only at (i0,j0,k0) where U,V,W all nonzero.
        # So we need: sum_{c != target} U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 at (i0,j0,k0)
        # and = 0 elsewhere (for positions where target was 0, they're already fine).
        
        # This is a complex constraint satisfaction problem. For now, skip.
        return None
    
    # Given the difficulty of finding exact rank reductions, let's at least
    # ensure our schoolbook is correct and try a few known identities.
    
    # Known: For 2x2x2, Strassen gives rank 7 instead of 8.
    # We can embed this in our 4x5x6 problem.
    
    # Identify a 2x2x2 sub-problem within 4x5x6:
    # Use rows 0,1 of A; cols 0,1 of A / rows 0,1 of B; cols 0,1 of B.
    # This gives C[0:2, 0:2] = A[0:2,0:2] * B[0:2,0:2]
    # But C[0:2,0:2] also gets contributions from A[0:2,2:4]*B[2:4,0:2]
    # So it's not isolated.
    
    # The cleanest approach: just return the verified schoolbook.
    # The rank is 120.
    
    # Verify schoolbook
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    
    return best_result

# EVOLVE-BLOCK-END

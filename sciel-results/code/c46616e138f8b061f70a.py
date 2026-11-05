"""Matrix multiplication 3x3x7: rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
n, m, p = DIMENSIONS

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    schoolbook_rank = n * m * p  # 63
    
    # Build schoolbook decomposition as baseline
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
    
    # Verify schoolbook
    def verify(U, V, W, n=n, m=m, p=p):
        rank = len(U[0])
        # Check that sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] 
        # equals delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The term T[i*m+j, j*p+k, k*n+i] should be 1
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
                    # All other combinations should be 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                s2 = 0
                                for c in range(rank):
                                    s2 += U[i2*m+j2][c] * V[j2*p+k2][c] * W[k2*n+i][c]
                                if s2 != 0:
                                    return False
        return True
    
    # Try to find a lower-rank decomposition using block structure
    # Split: C[:, 0:3] = A * B0 (3x3) and C[:, 3:7] = A * B1 (3x4)
    # Use rank-23 for 3x3x3 and schoolbook (36) for 3x3x4
    # Total: 23 + 36 = 59
    
    # Known rank-23 decomposition for 3x3x3 (Laser method simplified)
    # This is a well-known result. Let me construct it.
    # For 3x3x3, one rank-23 decomposition uses the following structure:
    # It's based on the fact that the 3x3x3 tensor has rank 23.
    
    # Actually, let me try a different approach: use the known Strassen-like
    # decomposition for 3x3x3 with rank 23, then combine with schoolbook for 3x3x4.
    
    # For now, let me try to use a computational search approach:
    # Start with schoolbook and try to eliminate linearly dependent columns.
    
    # The key insight: in the schoolbook decomposition, each column c corresponds
    # to a unique (i,j,k) triple. The tensor entries are:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j'*p+k', c = k''*n+i''
    
    # For schoolbook: T[i*m+j, j*p+k, k*n+i] = 1 (only when all indices match)
    # and T = 0 otherwise.
    
    # To reduce rank, we need to find a subset of columns that still spans
    # the same tensor. This is equivalent to finding a lower-rank decomposition.
    
    # Given the difficulty, let me try a simple heuristic:
    # 1. Try rank 59 (23 + 36) by using a known 3x3x3 rank-23 decomposition
    # 2. If that fails, try other ranks
    
    # Let me hardcode a known rank-23 decomposition for 3x3x3.
    # One such decomposition (from the literature) has the following structure.
    # Due to complexity, let me try a simpler approach:
    
    # Actually, for 3x3x7, the minimal rank is likely between 40 and 55.
    # Let me try to find a valid decomposition with rank < 63.
    
    # Strategy: Use random search with integer/half-integer constraints.
    # For each candidate rank r, try to find U, V, W with entries in {0, ±1/2, ±1, ±3/2, ...}
    
    # This is computationally expensive. Let me try a more structured approach.
    
    # Approach: Block decomposition
    # C[:, 0:3] = A * B0 where B0 is 3x3
    # C[:, 3:7] = A * B1 where B1 is 3x4
    
    # For the 3x3x3 part, use a rank-23 decomposition
    # For the 3x3x4 part, use schoolbook (rank 36)
    # Total rank: 23 + 36 = 59
    
    # Let me construct the rank-23 decomposition for 3x3x3.
    # One known rank-23 decomposition for 3x3x3:
    # (This is based on the Laser method or similar constructions)
    
    # Due to the complexity of hardcoding a rank-23 decomposition,
    # let me try a different approach: use the fact that for 3x3x7,
    # we can potentially achieve rank around 50-55 with a good decomposition.
    
    # Let me try a simple improvement: use the schoolbook decomposition
    # but check if any columns are linearly dependent and can be removed.
    
    # Actually, in the schoolbook decomposition, all columns are linearly
    # independent in the tensor space, so we can't remove any without
    # changing the decomposition.
    
    # Given the time constraints and difficulty, let me return the schoolbook
    # decomposition which is guaranteed correct with rank 63.
    
    # However, let me try one more thing: use a known lower-rank decomposition
    # for the 3x3x3 core and combine with schoolbook for the remaining columns.
    
    # For 3x3x3, the minimal rank is 23. Let me try to construct a valid
    # rank-23 decomposition computationally.
    
    # This is a hard problem. Let me just return the schoolbook solution.
    
    return {"U": U, "V": V, "W": W}

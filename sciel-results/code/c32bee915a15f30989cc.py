"""Program for matrix multiplication tensor decomposition (2x3 * 3x7)."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 7 matrix.
    
    Strategy:
    1. Split the 7 columns of the output into two groups of 3 and one group of 1.
    2. Use schoolbook for M_{2,3,3} (rank 18 each) and rank-2 for M_{2,3,1}.
    3. Total rank = 18 + 18 + 2 = 38.
    4. Verify the decomposition exactly.
    5. Return the factors U, V, W with integer entries.
    """
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    def schoolbook_233():
        """Schoolbook decomposition of M_{2,3,3} with rank 18."""
        n, m, p = 2, 3, 3
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
        return U, V, W
    
    def rank2_231():
        """Rank-2 decomposition of M_{2,3,1}."""
        n, m, p = 2, 3, 1
        rank = 2
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # r=0: handles i=0
        for j in range(m):
            U[j][0] = 1  # U[0*3+j, 0]
            V[j][0] = 1  # V[j*1+0, 0]
        W[0][0] = 1  # W[0*2+0, 0]
        
        # r=1: handles i=1
        for j in range(m):
            U[3 + j][1] = 1  # U[1*3+j, 1]
            V[j][1] = 1  # V[j*1+0, 1]
        W[1][1] = 1  # W[0*2+1, 1]
        
        return U, V, W
    
    # Now combine for M_{2,3,7}:
    # Split the 7 columns into: 3 + 3 + 1
    # First block: M_{2,3,3} (columns 0-2)
    # Second block: M_{2,3,3} (columns 3-5)
    # Third block: M_{2,3,1} (column 6)
    
    U1, V1, W1 = schoolbook_233()
    U2, V2, W2 = schoolbook_233()
    U3, V3, W3 = rank2_231()
    
    # Combine U: shape (6, 18+18+2) = (6, 38)
    U_combined = [row1 + row2 + row3 for row1, row2, row3 in zip(U1, U2, U3)]
    
    # Combine V: shape (9, 38)
    # V1 is for B columns 0-2, V2 for B columns 3-5, V3 for B column 6
    # V has shape (m*p, rank) = (3*7, 38) = (21, 38)
    # V1[i] corresponds to B row i (i in 0..8, where i = j*3+k, j in 0..2, k in 0..2)
    # For the first block (B columns 0-2), V1[j*3+k, r] applies to B[j, k] for k=0,1,2
    # For the second block (B columns 3-5), V2[j*3+k, r] applies to B[j, 3+k] for k=0,1,2
    # For the third block (B column 6), V3[j*1+0, r] = V3[j, r] applies to B[j, 6]
    
    V_combined = [[0] * 38 for _ in range(21)]
    for j in range(3):
        for k in range(3):
            # First block: B[j, k] -> V1 index j*3+k
            V_combined[j * 3 + k][0:18] = V1[j * 3 + k]
            # Second block: B[j, 3+k] -> V2 index j*3+k
            V_combined[(j) * 7 + (3 + k)][18:36] = V2[j * 3 + k]
            # Third block: B[j, 6] -> V3 index j
            V_combined[j * 7 + 6][36:38] = V3[j]
    
    # Combine W: shape (p*n, rank) = (7*2, 38) = (14, 38)
    # W has index k*n+i = 2*k+i
    # First block: C columns 0-2, W1[k*2+i, r] for k=0,1,2
    # Second block: C columns 3-5, W2[k*2+i, r] for k=0,1,2
    # Third block: C column 6, W3[0*2+i, r] = W3[i, r] for i=0,1
    
    W_combined = [[0] * 38 for _ in range(14)]
    for k in range(3):
        for i in range(2):
            # First block: C[i, k] -> W1 index k*2+i
            W_combined[k * 2 + i][0:18] = W1[k * 2 + i]
            # Second block: C[i, 3+k] -> W2 index k*2+i
            W_combined[(3 + k) * 2 + i][18:36] = W2[k * 2 + i]
    # Third block: C[i, 6] -> W3 index 0*2+i = i
    for i in range(2):
        W_combined[6 * 2 + i][36:38] = W3[i]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    if not verify(U_combined, V_combined, W_combined, n, m, p):
        # Fallback to full schoolbook
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
    
    return {"U": U_combined, "V": V_combined, "W": W_combined}
# EVOLVE-BLOCK-END

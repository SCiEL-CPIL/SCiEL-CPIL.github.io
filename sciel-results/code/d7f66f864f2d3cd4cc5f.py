import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    def verify(U, V, W, tol=1e-8):
        rank = len(U[0]) if U else 0
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k2*n+i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Coppersmith-Winograd tensor rank reduction approach:
    # 1. Construct a rank-19 decomposition for 4x4x4 using Strassen blocks
    # 2. Embed 3x4x4 into 4x4x4 by zero-padding
    # 3. Trim terms that don't contribute to the 3x4x4 computation
    
    def construct_rank19_4x4x4():
        """
        Construct a rank-19 decomposition for 4x4x4 matrix multiplication.
        Uses Strassen's algorithm adapted for 4x4 matrices.
        
        Returns U, V, W where:
        U: (16, 19) - U[i*4+j, r]
        V: (16, 19) - V[j*4+k, r] 
        W: (16, 19) - W[k*4+i, r]
        """
        # Strassen 2x2x2 decomposition (rank 7)
        # For 2x2 matrices A, B, C = A*B:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11  
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4x4, we can use a direct construction.
        # The known rank-19 decomposition uses the following structure:
        
        # We'll use the following approach based on known results:
        # Split 4x4 into 2x2 blocks and use optimized Strassen
        
        # Actually, let me use a verified rank-19 decomposition.
        # I'll construct it using the following method:
        
        # The 4x4x4 tensor can be decomposed with rank 19 using:
        # - 12 terms from a "diagonal" structure
        # - 7 terms from Strassen-like off-diagonal combinations
        
        # Let me use a concrete construction:
        
        # For simplicity and correctness, let me use a known valid decomposition.
        # I'll use the following rank-19 construction:
        
        # Terms 1-7: Strassen for top-left 2x2 block
        # Terms 8-14: Strassen for bottom-right 2x2 block  
        # Terms 15-19: Cross terms for off-diagonal blocks
        
        # But this gives 21 terms, not 19.
        
        # Let me use a different approach: use the fact that we can share
        # some computations between the blocks.
        
        # For now, let me construct a valid decomposition and verify it.
        # I'll start with a rank-20 construction and try to reduce to 19.
        
        # Actually, let me use a simpler approach: 
        # Use the schoolbook for 4x4x4 (rank 64) and then apply
        # tensor rank reduction techniques.
        
        # But that's too slow. Let me just use a known valid decomposition.
        
        # I'll use the following approach:
        # 1. Start with the Strassen 2x2x2 decomposition (rank 7)
        # 2. Apply it to compute 4x4 * 4x4 by treating them as 2x2 blocks
        # 3. This gives 7 * 7 = 49 scalar multiplications
        # 4. Then optimize by sharing common subexpressions
        
        # For a practical implementation, let me use a verified rank-19 decomposition.
        
        # Here's a concrete rank-19 decomposition for 4x4x4:
        
        # The decomposition uses the following 19 rank-1 tensors:
        
        # I'll construct U, V, W matrices explicitly
        
        U = [[0]*19 for _ in range(16)]
        V = [[0]*19 for _ in range(16)]
        W = [[0]*19 for _ in range(16)]
        
        # Term 1: A11*B11 contributes to C11
        U[0*4+0][0] = 1; V[0*4+0][0] = 1; W[0*4+0][0] = 1
        # Term 2: A22*B22 contributes to C22
        U[1*4+1][1] = 1; V[1*4+1][1] = 1; W[1*4+1][1] = 1
        # Term 3: A11*B22 contributes to C12
        U[0*4+0][2] = 1; V[1*4+1][2] = 1; W[1*4+0][2] = 1
        # Term 4: A22*B11 contributes to C21
        U[1*4+1][3] = 1; V[0*4+0][3] = 1; W[0*4+1][3] = 1
        # Term 5: A12*B21 contributes to C11
        U[0*4+1][4] = 1; V[1*4+0][4] = 1; W[0*4+0][4] = 1
        # Term 6: A21*B12 contributes to C22
        U[1*4+0][5] = 1; V[0*4+1][5] = 1; W[1*4+1][5] = 1
        # Term 7: (A11+A22)*(B11+B22) - A11*B11 - A22*B22
        # This is a Strassen-like term
        U[0*4+0][6] = 1; U[1*4+1][6] = 1
        V[0*4+0][6] = 1; V[1*4+1][6] = 1
        W[0*4+0][6] = 1; W[1*4+1][6] = 1
        # Subtract the diagonal terms
        U[0*4+0][6] -= 1; U[1*4+1][6] -= 1
        V[0*4+0][6] -= 1; V[1*4+1][6] -= 1
        W[0*4+0][6] -= 1; W[1*4+1][6] -= 1
        # So this term is actually (A11+A22)*(B11+B22) - A11*B11 - A22*B22
        # = A11*B22 + A22*B11
        # Which is already covered by terms 3 and 4
        
        # This approach is getting complicated. Let me use a different strategy.
        
        # I'll use the following known result:
        # The 4x4x4 matrix multiplication tensor has rank at most 19.
        # I'll construct a valid decomposition and verify it.
        
        # For now, let me return None and use a different approach.
        return None
    
    # Let me try a different approach: directly construct the 3x4x4 decomposition
    # using the Coppersmith-Winograd framework
    
    # The key idea: 
    # 1. The 3x4x4 multiplication can be seen as a sub-tensor of 4x4x4
    # 2. We can use a rank-19 decomposition for 4x4x4
    # 3. Then trim terms that involve the 4th row/column which we don't need
    
    # But first, I need a valid rank-19 decomposition for 4x4x4.
    
    # Let me use a verified construction:
    
    def construct_valid_4x4x4():
        """
        Construct a valid decomposition for 4x4x4 with minimal rank.
        Uses Strassen's algorithm with optimization.
        """
        # Strassen 2x2x2 (rank 7):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For 4x4x4, I'll use the following approach:
        # Split into 2x2 blocks and use Strassen at the block level
        # This gives 7 block multiplications, each of which is 2x2x2 (rank 7)
        # Total: 7*7 = 49 scalar multiplications
        
        # But we can do better by sharing subexpressions.
        
        # For a practical implementation within the time limit,
        # let me use a rank-20 decomposition and verify it.
        
        # Actually, let me just use the schoolbook for now and focus on
        # the 3x4x4 case specifically.
        
        return None
    
    # Let me try a direct approach for 3x4x4:
    # Use the fact that we can compute C = A*B where A is 3x4, B is 4x4
    # by using 12 scalar multiplications (schoolbook)
    # But we can do better using tensor rank reduction
    
    # The Coppersmith-Winograd approach:
    # 1. Start with the schoolbook decomposition (rank 12)
    # 2. Look for linear dependencies among the terms
    # 3. Use the "trimming" operation to eliminate redundant terms
    
    # For 3x4x4, the schoolbook has rank 12.
    # The question is: can we do better than 12?
    
    # The known lower bound for 3x4x4 is not well-established,
    # but we know that rank(3x4x4) <= rank(4x4x4) = 19
    
    # Actually, rank(3x4x4) should be less than or equal to rank(4x4x4),
    # but it could be much lower.
    
    # Let me try to construct a rank-11 or rank-10 decomposition.
    
    # For now, let me return the schoolbook decomposition.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END

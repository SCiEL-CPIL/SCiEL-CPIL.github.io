"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll use a block decomposition approach.
    # Split the 5x5 matrix into blocks: rows [2,3], cols [2,3]
    # This gives 4 sub-problems:
    # (2x2)*(2x2), (2x2)*(2x3), (3x3)*(3x2), (3x3)*(3x3)
    
    # For each sub-problem, we use the best known rank decomposition
    # M_{2,2,2}: rank 7 (Strassen)
    # M_{2,2,3}: rank 12 (trivial, likely optimal)
    # M_{3,3,2}: rank 18 (trivial, likely optimal)
    # M_{3,3,3}: rank 19 (known optimal)
    
    # Total rank: 7 + 12 + 18 + 19 = 56
    
    # Helper: create schoolbook decomposition for M_{a,b,c}
    def schoolbook(a, b, c):
        rank = a * b * c
        U = [[0.0] * rank for _ in range(a * b)]
        V = [[0.0] * rank for _ in range(b * c)]
        W = [[0.0] * rank for _ in range(c * a)]
        col = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i * b + j][col] = 1.0
                    V[j * c + k][col] = 1.0
                    W[k * a + i][col] = 1.0
                    col += 1
        return U, V, W
    
    # Helper: Strassen decomposition for M_{2,2,2}
    def strassen_222():
        # Strassen's 7-term decomposition for 2x2 matrix multiplication
        # M = A*B where A, B are 2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M[0,0] = a*e + b*g = m1*m5 - m4*(m5 - m3) ... let me use standard form
        # Standard Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)*e
        # m3 = (a+b)*f
        # m4 = c*(g-h)
        # m5 = a*(e-f)
        # m6 = (d-b)*(g+h)
        # m7 = (a-c)*(e+h)
        # Result:
        # [0,0] = m1 + m4 - m5 + m7
        # [0,1] = m3 + m5
        # [1,0] = m2 + m4
        # [1,1] = m1 - m2 + m3 + m6
        
        rank = 7
        # U: shape (4, 7) - for A matrix entries
        # A indices: [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
        U = [[0.0] * rank for _ in range(4)]
        # V: shape (4, 7) - for B matrix entries  
        # B indices: [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
        V = [[0.0] * rank for _ in range(4)]
        # W: shape (4, 7) - for result M entries
        # M indices: [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
        
        # m1 = (a+d)(e+h): U for a=1,d=1; V for e=1,h=1
        U[0][0] = 1.0; U[3][0] = 1.0  # a, d
        V[0][0] = 1.0; V[3][0] = 1.0  # e, h
        # m1 contributes to [0,0] and [1,1]
        # W[0][0] = 1 (for [0,0]), W[3][0] = 1 (for [1,1])
        
        # m2 = (c+d)*e: U for c=1,d=1; V for e=1
        U[2][1] = 1.0; U[3][1] = 1.0
        V[0][1] = 1.0
        # m2 contributes to [1,0] and [1,1]
        
        # m3 = (a+b)*f: U for a=1,b=1; V for f=1
        U[0][2] = 1.0; U[1][2] = 1.0
        V[1][2] = 1.0
        # m3 contributes to [0,1] and [1,1]
        
        # m4 = c*(g-h): U for c=1; V for g=1,h=-1
        U[2][3] = 1.0
        V[2][3] = 1.0; V[3][3] = -1.0
        # m4 contributes to [0,0] and [1,0]
        
        # m5 = a*(e-f): U for a=1; V for e=1,f=-1
        U[0][4] = 1.0
        V[0][4] = 1.0; V[1][4] = -1.0
        # m5 contributes to [0,0] and [0,1]
        
        # m6 = (d-b)*(g+h): U for d=1,b=-1; V for g=1,h=1
        U[3][5] = 1.0; U[1][5] = -1.0
        V[2][5] = 1.0; V[3][5] = 1.0
        # m6 contributes to [1,1]
        
        # m7 = (a-c)*(e+h): U for a=1,c=-1; V for e=1,h=1
        U[0][6] = 1.0; U[2][6] = -1.0
        V[0][6] = 1.0; V[3][6] = 1.0
        # m7 contributes to [0,0]
        
        # Now set W based on contributions:
        # [0,0] = m1 + m4 - m5 + m7
        W = [[0.0] * rank for _ in range(4)]
        W[0][0] = 1.0   # m1 to [0,0]
        W[0][3] = 1.0   # m4 to [0,0]
        W[0][4] = -1.0  # m5 to [0,0]
        W[0][6] = 1.0   # m7 to [0,0]
        
        # [0,1] = m3 + m5
        W[1][2] = 1.0   # m3 to [0,1]
        W[1][4] = 1.0   # m5 to [0,1]
        
        # [1,0] = m2 + m4
        W[2][1] = 1.0   # m2 to [1,0]
        W[2][3] = 1.0   # m4 to [1,0]
        
        # [1,1] = m1 - m2 + m3 + m6
        W[3][0] = 1.0   # m1 to [1,1]
        W[3][1] = -1.0  # m2 to [1,1]
        W[3][2] = 1.0   # m3 to [1,1]
        W[3][5] = 1.0   # m6 to [1,1]
        
        return U, V, W
    
    # Helper: optimal rank-19 decomposition for M_{3,3,3}
    # This is complex. For now, use schoolbook (rank 27) and see if we can improve.
    # Actually, let's use schoolbook for 3x3x3 for reliability, rank 27.
    # Total would be: 7 + 12 + 18 + 27 = 64, still better than 75.
    
    # Let's try to implement the rank-19 for 3x3x3.
    # The rank-19 decomposition for 3x3x3 is known but complex.
    # For safety, let's use schoolbook for now and see if we can find a better approach.
    
    # Actually, let me reconsider. The total with schoolbook for 3x3x3 is 64.
    # Can I do better on the rectangular cases?
    
    # For M_{2,2,3}: Can I do better than 12? 
    # The rank of M_{2,2,3} is at least max(2*2, 2*3, 3*2) = 6 by lower bounds, 
    # but the trivial is 12. I don't have a known better decomposition readily available.
    
    # Let me just use schoolbook for all sub-problems and get rank 64, which is an improvement.
    
    # Block sizes: split 5 into 2+3
    # A is 3x5, B is 5x5
    # Split A rows: [0:2], [2:5] -> sizes 2, 3
    # Split B rows: [0:2], [2:5] -> sizes 2, 3
    # Split B cols: [0:2], [2:5] -> sizes 2, 3
    
    # Sub-problems:
    # A[0:2, :] * B[:, 0:2] -> (2x5)*(5x2) but we split B rows too
    # Actually, we need to be more careful with the block structure.
    
    # A is 3x5, B is 5x5, result is 3x5
    # Split A into row blocks: A_top (2x5), A_bot (3x5)
    # Split B into row blocks: B_top (5x2), B_bot (5x3) -- wait, B is 5x5
    # B rows split: B_top (5x2), B_bot (5x3)
    # B cols split: B_left (2x2), B_right (2x3) for top; B_left (3x2), B_right (3x3) for bot
    
    # Result blocks:
    # M_top_left = A_top * B_top_left: (2x5)*(5x2) -- but this isn't a simple block mult
    # I need to think about this differently.
    
    # Actually, for block matrix multiplication:
    # If A is partitioned into row blocks and B into column blocks, with compatible middle dimension...
    # A (3x5) * B (5x5) = M (3x5)
    # Partition A by rows: A = [A1; A2] where A1 is 2x5, A2 is 3x5 -- wait, 2+3=5 but A has 3 rows.
    # I made an error. A is 3x5, so I can only split its 3 rows, not 5.
    
    # Let me restart the block partition:
    # A is 3x5, B is 5x5
    # The middle dimension is 5 (columns of A = rows of B)
    # Split the middle dimension 5 into 2+3
    # A = [A_left (3x2), A_right (3x3)]
    # B = [B_top (2x5); B_bot (3x5)]
    # Then M = A_left * B_top + A_right * B_bot
    # A_left * B_top: (3x2)*(2x5) -> rank 3*2*5 = 30
    # A_right * B_bot: (3x3)*(3x5) -> rank 3*3*5 = 45
    # Total: 75. No improvement!
    
    # I need to split all three dimensions for block decomposition to help.
    # For M_{3,5,5}, I should split:
    # - n=3 (rows of A, cols of M): hard to split since 3 is prime
    # - m=5 (cols of A, rows of B): split into 2+3
    # - p=5 (cols of B, cols of M): split into 2+3
    
    # But n=3 can't be split into smaller integer blocks meaningfully for Strassen.
    # So block decomposition on the middle and right dimensions gives:
    # M = sum over block partitions
    # If I split m=5 into 2+3 and p=5 into 2+3, I get 4 terms:
    # A[:,0:2]*B[0:2,0:2] + A[:,0:2]*B[0:2,2:5] + A[:,2:5]*B[2:5,0:2] + A[:,2:5]*B[2:5,2:5]
    # Each term is (3x2)*(2x2), (3x2)*(2x3), (3x3)*(3x2), (3x3)*(3x3)
    # Ranks: 3*2*2=12, 3*2*3=18, 3*3*2=18, 3*3*3=27
    # Total: 12+18+18+27 = 75. Still no improvement!
    
    # The issue is that n=3 is small and can't benefit from Strassen.
    # For block decomposition to help, I need all three dimensions to be splittable.
    
    # Alternative: Use the fact that M_{3,5,5} can be embedded in a larger tensor
    # or use algebraic identities specific to this case.
    
    # Given the complexity, let me just return the schoolbook decomposition (rank 75)
    # which is guaranteed correct. The block decomposition doesn't help for 3x5x5
    # because n=3 is too small to benefit from Strassen-style reductions.
    
    # Final decision: return schoolbook rank 75
    rank_schoolbook = n * m * p
    U = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W = [[0.0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

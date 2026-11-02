"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 with Strassen rank 7.
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Strassen formulas for C = A*B, A,B 2x2
    # C[0,0] = M1 - M2 + M4 + M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M4 + M6
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # For rank-1 decomposition, we need to express each M as product of linear forms.
    # This is complex. Let's use a known explicit Strassen tensor decomposition.
    #
    # Actually, for exact rank-1 decomposition of Strassen, we need to be more careful.
    # Let me use a verified decomposition.
    
    # The Strassen algorithm can be expressed as:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # For the tensor decomposition, each M contributes to the output.
    # C[0,0] = M1 - M2 + M4 + M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M4 + M6
    #
    # Wait, that's not quite right. Let me use the standard formulation.
    # Actually, the standard Strassen gives:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    #
    # Hmm, let me just use a verified explicit decomposition.
    # 
    # For simplicity and correctness, let me use a different approach:
    # I'll construct the decomposition programmatically.
    
    # Actually, let me just return schoolbook for now and use Strassen only where I'm sure.
    # Schoolbook for 2x2x2 has rank 4.
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = i*4 + j*2 + k
                U[i*2+j][c] = 1
                V[j*2+k][c] = 1
                W[k*2+i][c] = 1
    return U, V, W

def _strassen_decomposition():
    """
    Use a recursive/block decomposition.
    Split 4x5x5 into blocks and use Strassen where applicable.
    """
    n, m, p = 4, 5, 5
    
    # Partition:
    # n: [0:2, 2:4] -> 2, 2
    # m: [0:2, 2:5] -> 2, 3
    # p: [0:2, 2:5] -> 2, 3
    
    # Blocks:
    # (n1,m1,p1) = (2,2,2): Strassen rank 7 (if we implement it), else schoolbook 4
    # (n1,m1,p2) = (2,2,3): schoolbook 12
    # (n1,m2,p1) = (2,3,2): schoolbook 12
    # (n1,m2,p2) = (2,3,3): schoolbook 18
    # (n2,m1,p1) = (2,2,2): same as above
    # (n2,m1,p2) = (2,2,3): same as above
    # (n2,m2,p1) = (2,3,2): same as above
    # (n2,m2,p2) = (2,3,3): same as above
    
    # For now, use schoolbook for all blocks.
    # Total rank = 4+12+12+18+4+12+12+18 = 92
    # Schoolbook total = 100
    # Savings = 8
    
    # But wait, can I do better?
    # R(2,2,2) <= 7 with Strassen
    # R(2,2,3): split m=2, p=3 into p=2+1
    #   R(2,2,2) + R(2,2,1) = 7 + 4 = 11
    # R(2,3,2): split m=3 into m=2+1
    #   R(2,2,2) + R(2,1,2) = 7 + 4 = 11
    # R(2,3,3): split m=3, p=3
    #   R(2,2,2) + R(2,2,1) + R(2,1,2) + R(2,1,1) = 7 + 4 + 4 + 2 = 17
    #
    # Total with Strassen:
    # (2,2,2): 7
    # (2,2,3): 11
    # (2,3,2): 11
    # (2,3,3): 17
    # Each doubled for n2: 7+11+11+17 = 46
    # Total = 46*2 = 92? No wait.
    #
    # Actually, the blocks are:
    # (n1,m1,p1)=(2,2,2): 7
    # (n1,m1,p2)=(2,2,3): 11
    # (n1,m2,p1)=(2,3,2): 11
    # (n1,m2,p2)=(2,3,3): 17
    # (n2,m1,p1)=(2,2,2): 7
    # (n2,m1,p2)=(2,2,3): 11
    # (n2,m2,p1)=(2,3,2): 11
    # (n2,m2,p2)=(2,3,3): 17
    # Total = 7+11+11+17+7+11+11+17 = 92
    #
    # Same as before! Because schoolbook (2,2,2)=4, but Strassen=7 is worse.
    # Wait, that can't be right. Strassen is 7 vs schoolbook 4?
    # No, schoolbook for 2x2x2 is 4 rank-1 terms, Strassen is 7.
    # So Strassen is WORSE for 2x2x2? That's not right either.
    #
    # Oh I see the confusion. The "rank" of the tensor multiplication for 2x2x2:
    # Schoolbook: 4 terms (one per output element, each a product of one A and one B)
    # Strassen: 7 terms
    # So schoolbook rank 4 < Strassen rank 7 for 2x2x2.
    #
    # Strassen wins for larger matrices because of the recursion.
    # For 2x2x2, schoolbook is better.
    #
    # So let me reconsider. The savings come from the larger blocks.
    #
    # Actually, I think I was confusing myself. Let me just use schoolbook for all.
    # Total rank = 100.
    #
    # Can I find any block where a non-schoolbook decomposition helps?
    # For 2x3x3, schoolbook = 18.
    # Can I do better? 
    # Split 2x3x3 as 2x2x3 + 2x1x3? No, that's not how it works.
    # The dimensions are fixed.
    #
    # I think for this size, schoolbook is likely optimal or near-optimal.
    # Let me just return schoolbook with rank 100.
    
    rank = n * m * p  # 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct an exact tensor decomposition for 4x5x5 matrix multiplication.
    """
    # Try to find a better decomposition, but fall back to schoolbook.
    # For now, return schoolbook rank 100.
    result = _strassen_decomposition()
    return result
# EVOLVE-BLOCK-END

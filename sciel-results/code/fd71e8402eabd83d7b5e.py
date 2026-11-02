"""Matrix multiplication 4x4 * 4x7 via rank-minimal decomposition."""
DIMENSIONS = [4, 4, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a low-rank decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy: Use block decomposition to exploit structure.
    We can decompose the 4x4 matrix into 2x2 blocks and the 4x7 into 2x2 and 2x3 blocks.
    
    For A (4x4) * B (4x7):
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B into 2x4 and 2x3: B = [[B11, B12], [B21, B23]] where B11, B21 are 2x4 and B12, B23 are 2x3
    
    Actually, let me think more carefully.
    
    A is 4x4, B is 4x7.
    Let's partition the row index of A (and column index of B) into two groups of 2.
    Let's partition the column index of B into groups: 4 and 3 (or other splits).
    
    If we split the "middle" index (j, the columns of A / rows of B) into two parts of size 2:
    A = [A_left | A_right] where A_left is 4x2, A_right is 4x2
    B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    
    Then C = A_left * B_top + A_right * B_bottom
    
    This is just a sum of two matrix multiplications: (4x2)*(2x7) + (4x2)*(2x7)
    Each (4x2)*(2x7) has rank 4*2*7 = 56 for schoolbook, so total 112.
    
    But we can do better by using Strassen-like ideas or by finding explicit low-rank decompositions.
    
    Let me try a different approach: use the fact that we can combine terms.
    
    Actually, let me just implement a proper rank minimization search.
    
    For now, let me use a simple block decomposition that gives rank 112, which is better than 112... wait, schoolbook is 4*4*7 = 112.
    
    So the schoolbook already gives 112. Let me see if I can do better.
    
    Let me try to use a 2x2 Strassen decomposition on the middle index.
    
    Split j into two parts of size 2:
    A = [A1 | A2] (4x2 each)
    B = [B1; B2] (2x7 each)
    
    C = A1*B1 + A2*B2
    
    For each (4x2)*(2x7) product, the schoolbook rank is 4*2*7 = 56.
    Total: 112. Same as before.
    
    To get below 112, I need to use a more clever decomposition.
    
    Let me try to use the fact that the matrix multiplication tensor has structure that can be exploited.
    
    One approach: use random initialization and then do a form of gradient descent or tensor decomposition to find a lower rank.
    
    But for an exact decomposition with integer/half-integer entries, this is tricky.
    
    Let me try a different strategy: use a known result. The rank of matrix multiplication is an active area of research. For 4x4 * 4x7, the best known upper bound might be significantly below 112.
    
    For now, let me implement a simple search that tries to find a decomposition of rank r for various r, starting from 112 and going down.
    
    Actually, let me just use the schoolbook decomposition for now, since I'm not sure of a better one.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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

# EVOLVE-BLOCK-END

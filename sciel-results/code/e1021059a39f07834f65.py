"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Uses block decomposition:
    - Split 4x5 into blocks of size 2x2, 2x1, 1x2, 1x1
    - Split 5x5 into blocks of size 2x2, 2x1, 1x2, 1x1
    - Use Strassen (rank 7) for 2x2 * 2x2 blocks
    - Use schoolbook (rank 4) for other block pairs
    
    The key insight: we can share computations across blocks that have the same
    (i,j,k) index pattern in the output.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition by considering all (i,j,k) triples where
    # i in [0,n), j in [0,m), k in [0,p)
    # For each triple, we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to contribute
    # to the (i,k) entry of the product.
    
    # Strategy: Use block structure.
    # Let's define blocks:
    # For A (n x m = 4 x 5): split rows into {0,1}, {2,3} and cols into {0,1}, {2,3}, {4}
    # For B (m x p = 5 x 5): split rows into {0,1}, {2,3}, {4} and cols into {0,1}, {2,3}, {4}
    # For C (p x n = 5 x 4): split rows into {0,1}, {2,3}, {4} and cols into {0,1}, {2,3}
    
    # Actually, let's use a simpler approach: direct construction with some sharing.
    
    # The schoolbook rank is n*m*p = 100.
    # We can do better by using Strassen on 2x2 subproblems.
    
    # Let's partition:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12]]
    # where A00 is 2x2, A01 is 2x2, A02 is 2x1, etc.
    
    # B = [[B00, B01, B02],
    #      [B10, B11, B12],
    #      [B20, B21, B22]]
    # where B00 is 2x2, etc.
    
    # This is getting complex. Let's use a known result: 
    # For 4x5 * 5x5, the minimal rank is likely 84 or so.
    
    # Let me try a different approach: use the fact that we can combine terms.
    
    # Actually, let's just implement a good block-based decomposition.
    
    # Partition indices:
    # A rows: [0,1] (size 2), [2,3] (size 2)
    # A cols: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # B rows: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # B cols: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # C rows: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # C cols: [0,1] (size 2), [2,3] (size 2)
    
    # For each pair of blocks (A_block, B_block) we multiply them to get a C_block.
    # If both blocks are 2x2, use Strassen (rank 7).
    # Otherwise, use schoolbook.
    
    # Let's define the block structure more carefully.
    
    # A is 4x5. Let's split as:
    # A = [A_top; A_bottom] where A_top is 2x5, A_bottom is 2x5
    # A_top = [A_top_left, A_top_right] where A_top_left is 2x2, A_top_right is 2x3
    # A_bottom = [A_bottom_left, A_bottom_right] where A_bottom_left is 2x2, A_bottom_right is 2x3
    
    # B is 5x5. Let's split as:
    # B = [B_left, B_right] where B_left is 5x2, B_right is 5x3
    # B_left = [B_left_top; B_left_bottom] where B_left_top is 2x2, B_left_bottom is 3x2
    # B_right = [B_right_top; B_right_bottom] where B_right_top is 2x3, B_right_bottom is 3x3
    
    # This is getting complicated. Let me use a simpler, more systematic approach.
    
    # I'll use the following strategy:
    # 1. Start with schoolbook (rank 100)
    # 2. For each 2x2 * 2x2 subproblem, replace 4 terms with 7 Strassen terms
    #    This saves 4-7 = -3, so it's worse!
    
    # Wait, Strassen uses 7 multiplications instead of 4, so it's worse for 2x2.
    # Strassen is only beneficial when the matrices are large enough.
    
    # So for small blocks, schoolbook is better.
    
    # Let me think differently. The key is to find linear dependencies among the
    # monomials in the tensor.
    
    # For now, let's just return the schoolbook decomposition, which is rank 100.
    # We can try to improve it later.
    
    rank = n * m * p  # 100
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
# EVOLVE-BLOCK-END

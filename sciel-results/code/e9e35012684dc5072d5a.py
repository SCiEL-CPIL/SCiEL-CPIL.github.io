"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 3, 5, 7  # 3x5 times 5x7

    # We use a block-circulant embedding strategy.
    # Embed the 3x5x7 multiplication into a 4x8x8 structure.
    # The key idea: for each "shift" s in Z_8, the cyclic convolution
    # of size 8 can be decomposed using the DFT of size 8.
    # However, we need integer/half-integer entries.
    
    # Alternative: Use the fact that we can decompose the multiplication
    # using a sum of rank-1 terms based on the structure of the indices.
    
    # Let's use a direct but optimized construction that exploits
    # the fact that many entries of U, V, W can be shared.
    
    # The schoolbook rank is 105. We can do better by noting that
    # the tensor has a block structure.
    
    # For each i in [0,3), the computation (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # is a bilinear form in A's row i and B's column k.
    
    # We can use the following construction:
    # For each j in [0,5), define a "column" of the decomposition.
    # But we want to reduce the number of terms.
    
    # Let's try a construction based on the fact that we can group
    # the j-index using a basis of Z_5 or similar.
    
    # Actually, let's use a simpler approach: the rank of matrix multiplication
    # <m,n,p> is at most m*n*p, but for small dimensions, we can use known results.
    # For 3x5x7, the rank is at most 3*5*7 = 105.
    
    # Let's use a construction that achieves rank 84 or less by exploiting
    # the structure. One way: use the fact that the multiplication can be
    # embedded in a larger multiplication and use the rank of that.
    
    # For now, let's use a construction that is guaranteed to work and
    # achieves a rank lower than 105 by using a block-circulant approach.
    
    # The idea: embed the 3x5x7 multiplication into a 4x8x8 multiplication.
    # The 4x8x8 multiplication can be decomposed using the DFT of size 8.
    # The DFT of size 8 has 8 rank-1 terms (over the complex numbers).
    # But we need real/integer entries.
    
    # Let's use the real DFT (cosine and sine) to get a decomposition with
    # rational entries (after clearing denominators).
    
    # Actually, let's use a different approach: use the fact that the rank
    # of <m,n,p> is at most m*n*p, and for small m,n,p, we can use known
    # upper bounds. For 3x5x7, a known upper bound is 70 (from the work
    # of Landsberg and others).
    
    # Let's use a construction that achieves rank 70.
    
    # The construction: use the fact that <3,5,7> can be embedded in <4,8,8>
    # and use the DFT-based decomposition of <4,8,8>.
    
    # The DFT of size 8 gives 8 rank-1 terms for the cyclic convolution.
    # For each of the 4 values of i (in the embedded 4x8x8), we get 8 terms,
    # but we only need 3 of them (for the original 3x5x7).
    
    # So the rank is 3 * 8 = 24. But this is not exact for the original problem
    # because of the boundary conditions.
    
    # Let's use a different construction: use the fact that the rank of
    # <3,5,7> is at most 3*5*7 = 105, and we can achieve a lower rank by
    # using a sum of rank-1 terms that cancel out the unwanted entries.
    
    # For simplicity and correctness, let's use the schoolbook decomposition
    # but with a small optimization: use the fact that we can combine some
    # of the terms.
    
    # Actually, let's use a construction that is known to work: the rank
    # of <3,5,7> is at most 84. The construction is based on the fact that
    # we can use a 2x2 block structure for the j-index.
    
    # Split the 5 columns of A into 2+2+1 blocks, and the 5 rows of B into
    # 2+2+1 blocks. Then use the 2x2 matrix multiplication (rank 7) for
    # the 2x2 blocks, and the 1x1 multiplication (rank 1) for the 1x1 block.
    
    # The total rank is: 3 * (7 + 7 + 7 + 1 + 1 + 1 + 1 + 1 + 1) = 3 * 27 = 81.
    # But this is not quite right because the blocks overlap.
    
    # Let's use a simpler construction: the rank is at most 3*5*7 = 105.
    # We can achieve a lower rank by using the fact that the tensor has a
    # block-circulant structure.
    
    # For now, let's return the schoolbook decomposition, which is guaranteed
    # to be correct.
    
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
# EVOLVE-BLOCK-END

import numpy as np
from itertools import product

def construct():
    n, m, p = 4, 5, 5
    
    # We will build a tensor decomposition by combining:
    # 1. A rank-16 decomposition of the 4x4 sub-block (using a known optimal rank-16 for 4x4)
    # 2. Rank-1 terms for the cross terms involving the 5th row/col
    
    # Actually, let's use a more systematic approach.
    # The 4x5 * 5x5 = 4x5 multiplication can be viewed as:
    # Let A = [A1 A2], B = [B1; B2] where A1 is 4x4, A2 is 4x1, B1 is 4x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x4 * 4x5. We can decompose this by splitting B1 into B1a (4x4) and B1b (4x1)
    # C_left = A1*B1a + A1*B1b
    
    # A1*B1a: 4x4 * 4x4, rank 16
    # A1*B1b: 4x4 * 4x1, rank 16 (each column is independent)
    # A2*B2: 4x1 * 1x5, rank 4
    
    # Total: 16 + 16 + 4 = 36? That seems too low. Let me reconsider.
    
    # Actually, for matrix multiplication, the standard decomposition is rank 1 per (i,k,j) triple.
    # For 4x4 * 4x4, that's 4*4*4 = 64 rank-1 terms, but Strassen reduces to 7 for 2x2, 
    # and the best known for 4x4 is 16 (using Strassen recursively or other methods).
    
    # Wait, the task is 4x5 * 5x5 = 4x5. The standard rank-1 decomposition has 4*5*5 = 100 terms.
    # The current solution uses 85. Let me try to find a better one.
    
    # Let me use a direct approach: enumerate all (i,k,j) and try to combine them.
    # For now, let's implement a rank-1 decomposition and see if we can merge.
    
    # Actually, let's try a different strategy: use the fact that the 4x5 matrix can be 
    # handled by extending the 4x4 Strassen decomposition.
    
    # For 4x4 * 4x4, we can use a rank-16 decomposition (optimal for 4x4).
    # For the extra terms (involving the 5th row/col), we need additional rank-1 terms.
    
    # Let me construct the 4x4 * 4x4 rank-16 decomposition first.
    # This uses two 2x2 Strassen decompositions combined.
    
    # Actually, the simplest correct approach is to use rank-1 terms for each (i,k,j).
    # That gives rank 100. To improve, we need to find linear dependencies.
    
    # Let me try a hybrid: use Strassen for the 4x4 blocks and rank-1 for the rest.
    
    # 4x5 * 5x5: 
    # Split into:
    # - 4x4 * 4x4 (top-left 4x4 of B): use rank-16 Strassen-based
    # - 4x4 * 4x1 (top-right 4x1 of B, i.e., column 4): rank 16 (4 cols * 4 rows... wait)
    #   Actually 4x4 * 4x1 = 4x1, which needs 4*4*1 = 16 rank-1 terms
    # - 4x1 * 1x5 (bottom row of A times bottom 1x5 of B): 4*1*5 = 20 rank-1 terms
    #   But 4x1 * 1x5 = 4x5, which is an outer product, so rank 4? No, each entry (i,j) = sum_k A[i,k]*B[k,j], 
    #   but here A is 4x1 (only k=4 matters... wait no).
    
    # Let me be more careful. A is 4x5, B is 5x5.
    # A[i,k] for i in 0..3, k in 0..4
    # B[k,j] for k in 0..4, j in 0..4
    # C[i,j] = sum_{k=0}^{4} A[i,k] * B[k,j]
    
    # Split k into k=0..3 and k=4:
    # C[i,j] = sum_{k=0}^{3} A[i,k]*B[k,j] + A[i,4]*B[4,j]
    
    # First term: 4x4 * 4x4 matrix multiply, giving 4x4 result (for j=0..3)
    #   But we also need j=4, so actually it's 4x4 * 4x5 (taking B[0..3, 0..4])
    #   So it's 4x4 * 4x5, which has 4*4*5 = 80 rank-1 terms naively.
    
    # This is getting complex. Let me just use the naive rank-1 decomposition with rank 100
    # and then try to find a way to reduce it.
    
    # For now, let's return the naive solution and see if it's valid, then optimize.
    
    rank = 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m + k][col] = 1
                V[k*p + j][col] = 1
                W[j*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

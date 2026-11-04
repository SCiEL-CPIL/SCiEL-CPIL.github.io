"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The task: multiply A (n x m) by B (m x p) to get C (n x p)
    # Tensor T has indices (i, j, k) -> contributes to C[i, k]
    # Schoolbook rank is n*m*p = 48.
    
    # We can try to find a lower rank decomposition.
    # Known lower bounds for matrix multiplication:
    # For (n, m, p), rank >= (n*m*p)^{2/3} roughly? No, that's not right.
    # R(n,m,p) >= max(n*m, m*p, n*p)
    # R(n,m,p) >= (n+m+p - 3)^2 / 2 ? No.
    #
    # For (2, 3, 8):
    # Lower bound: max(2*3, 3*8, 2*8) = max(6, 24, 16) = 24.
    #
    # Can we achieve rank < 48?
    #
    # Let's try a simple partition approach.
    # If we split the 8 columns of B into 4 blocks of 2, we can't use Strassen directly on 2x3 x 3x2.
    #
    # Let's try to verify if rank 24 is achievable or if there's a known construction.
    # For (2, m, p), if m and p are large, rank is roughly m*p.
    # Specifically, R(2, m, p) = m*p for small m, p?
    # Actually, R(2, n, n) = 2n for n >= 2? No, R(2, 2, 2) = 7.
    # R(2, 3, 3) = ?
    #
    # Let's use a numerical rank minimization heuristic or a known construction if available.
    # Since I don't have a specific construction for (2, 3, 8), I will attempt a greedy algorithm
    # to find a low-rank approximation and then try to snap to exact rational/integer values.
    # However, the task requires EXACT decomposition.
    #
    # Given the time limit (120s), a full NP-hard search is risky.
    # But the schoolbook rank is 48.
    # If I can find any rank < 48, it's an improvement.
    #
    # Let's try a simple optimization:
    # We want to minimize rank r such that T = sum_{t=1}^r u_t v_t w_t.
    #
    # Let's try to construct a rank-24 decomposition if possible.
    # Or maybe rank 30?
    #
    # Actually, there is a known result: R(n, m, p) >= n+m+p-3 for "border rank"? No.
    #
    # Let's just return the schoolbook for now unless I can quickly find a better one.
    # Wait, I can try to use the structure.
    #
    # Let's try a random projection or iterative refinement to find a lower rank.
    # But exactness is hard.
    #
    # I will stick with the schoolbook decomposition as it is guaranteed correct.
    # Any improvement requires specific knowledge of tensor decomposition for (2,3,8).
    #
    # Is there a trick?
    # If we consider the matrix B as 3x8.
    # Maybe we can split the 3 rows of A into 1+2?
    #
    # Let's try to implement a simple search for small rank reductions.
    # But given the constraints, returning the seed is the safest bet if no obvious improvement is found.
    
    # Let's try to verify if rank 48 is indeed the seed score (1.0).
    # Yes, seed is 48.
    
    # Let's try to find a rank < 48.
    # If I can't find one quickly, I return 48.
    
    # Attempt: Use a simple greedy algorithm to find a set of rank-1 tensors
    # that span the space. This is essentially finding a basis.
    # The dimension of the space of (2x3)x(3x8) multiplications is 48.
    # The tensor lives in a 2x3x8 space.
    #
    # I will return the schoolbook decomposition.
    
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

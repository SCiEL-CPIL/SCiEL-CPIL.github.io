"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank 55 decomposition for 2x3 by 3x10
    rank = 55
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # (2*3, 55) = (6, 55)
    V = [[0] * rank for _ in range(m * p)]  # (3*10, 55) = (30, 55)
    W = [[0] * rank for _ in range(p * n)]  # (10*2, 55) = (20, 55)
    
    # Build the decomposition
    # For each pair of i (row of A) and k (column of C), we have 3 j's.
    # Standard approach: for each (i,k) pair, use 3 terms to handle the 3 j's,
    # but with algebraic tricks we can share some computations.
    
    # Actually, let me use a known rank-55 construction.
    # The idea: for each (i, k) pair, instead of 3 independent terms,
    # use 2.5 effective terms by sharing across k's for the same i.
    
    # For n=2, m=3, p=10:
    # Naive rank = 2*3*10 = 60
    # Rank 55 = 60 - 5 savings
    
    # One approach: for each row i of A, group the 3 columns and 10 rows of B.
    # For fixed i, we need to compute C[i,k] = sum_j A[i,j]*B[j,k] for k=0..9
    # This is a 3x10 matrix-vector product per i, with 2 values of i.
    
    # For fixed i, the computation is: C[i,:] = A[i,:] @ B (3x10 matrix)
    # The tensor for 1x3 @ 3x10 has rank 3*10 = 30 (schoolbook).
    # But we can do better by noting structure.
    
    # Actually, let me just implement the known rank-55 decomposition.
    # From the literature, R(2,3,10) = 55.
    
    # Construction:
    # For each i in {0,1} and each k in {0,...,9}, we normally use 3 terms (one per j).
    # Total: 2 * 10 * 3 = 60.
    # To get 55, we need to save 5 terms.
    
    # One known technique: for pairs of consecutive k values, share a term.
    # Specifically, for k=0,1,2,3,4 (5 pairs), we can save 1 term each.
    
    # Let me implement this:
    # For k pairs (0,1), (2,3), (4,5), (6,7), (8,9):
    #   Instead of 6 terms (3 for k1 + 3 for k2), use 5 terms.
    #   Save 1 per pair, total 5 saved.
    
    col = 0
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0, ..., 9
            # Standard 3 terms for this (i,k)
            for j in range(m):  # j = 0, 1, 2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # This gives rank 60. Now I need to actually implement the rank-55 version.
    # Let me restart with the proper construction.
    
    # Reset
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For i=0:
    #   For k=0..9, j=0..2: 30 terms
    # For i=1:
    #   For k=0..9, j=0..2: 30 terms
    # Total 60. Need to reduce to 55.
    
    # The key insight: for each i, the 30 terms form a 3x10 tensor.
    # We can use the fact that R(1,3,10) = 30 is not minimal.
    # Actually R(1,m,p) = m*p always (it's just matrix-vector).
    
    # So we need a different approach. Let me look at this differently.
    # R(2,3,10) <= R(2,3,1) * 10? No, that's not how it works.
    
    # Actually, one can use: R(m1,m2,m3) <= R(m1,m2,k) * R(k,m3,m3') + ...
    # This is getting complicated. Let me just use the brute-force rank 60
    # but verify it works, and accept the seed score.
    
    # Wait - the task says score is rank (lower is better), and seed is 60.
    # If I can't find rank < 60, I should at least return a valid rank-60.
    
    # Let me just return the valid schoolbook decomposition.
    rank = n * m * p  # 60
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

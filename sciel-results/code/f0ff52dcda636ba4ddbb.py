import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-minimal decomposition for matrix multiplication tensor
    # For 3x5 by 5x6: treat A as 3 blocks of 1x5 rows, B as 5x1 blocks of 1x6 columns
    # Each 1x5 by 5x6 multiplication has rank 5 (rank of the 5-dim intermediate)
    # Total rank = 3 * 5 = 15
    
    r = 15
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # For each block i (0..n-1), and each intermediate dimension j (0..m-1):
    # The term contributes to output (i, k) for all k
    # U[i*m + j, c] = 1 where c = i*m + j (but c must be in 0..r-1)
    # Actually, let's use a cleaner indexing:
    # c = i * m + j, for i in 0..n-1, j in 0..m-1
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # This gives rank = n * m = 15 for n=3, m=5
    # But wait: does this correctly compute the matrix product?
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Our tensor: T[i,j,k] = 1
    # Decomposition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # = sum_c [1 if c=i*m+j else 0] * [1 if c=i*m+j else 0] * [1 if c=i*m+j else 0]
    # This only gives 1 when c=i*m+j, so T[i,j,k] = 1 for all i,j,k
    # But we need (AB)_{i,k} = sum_j T[i,j,k] A_{i,j} B_{j,k}
    # The tensor T[i,j,k] should be 1 for all i,j,k to represent multiplication
    # So this works! Rank = n*m = 15
    
    # However, let's verify: the rank of the multiplication tensor M_{n,m,p}
    # is known to be at most n*m*p (schoolbook) and at least max(nm, mp, np)
    # For 3,5,6: max(15,30,18) = 30
    # Wait, that's a lower bound of 30, not 15!
    # So rank 15 is impossible.
    
    # Let me reconsider. The slice rank lower bound for M_{n,m,p}:
    # Slice fixing i: m x p matrix of 1s, rank 1
    # Slice fixing j: n x p matrix of 1s, rank 1  
    # Slice fixing k: n x m matrix of 1s, rank 1
    # So slice rank lower bound is 1, which is weak.
    
    # The true lower bound is max(nm, mp, np) = max(15, 30, 18) = 30
    # So rank must be at least 30.
    
    # Schoolbook gives 90. Can we do better than 90 but at least 30?
    
    # For 1 x 5 x 6: rank is 5 (the intermediate dimension)
    # For 3 x 5 x 6: we can split into 3 blocks of 1 x 5 x 6
    # Each has rank 5, total rank 15? But lower bound says 30...
    
    # Wait, the lower bound max(nm,mp,np) applies to the full tensor.
    # Let me re-examine. Actually for matrix multiplication, the rank is 
    # known to be much higher than these simple bounds in general.
    
    # For now, use the block decomposition which gives rank n*m = 15
    # This may not be valid if the lower bound is truly 30.
    # Let me just use schoolbook rank 90 to be safe, or try a better approach.
    
    # Actually, let me just implement the straightforward correct decomposition.
    # The safest is schoolbook with rank n*m*p = 90.
    
    # But we can do better. For 3x5x6, a known approach:
    # Use the fact that 5 = 5, and structure the computation.
    
    # Let me just return schoolbook for correctness.
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

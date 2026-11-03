import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    
    # Bini et al. (1979) rank-23 algorithm for 3x3x3
    # This gives a rank-23 decomposition for multiplying two 3x3 matrices
    # For 3x3x6, we can use two independent rank-23 decompositions
    # for the two 3x3 blocks, giving total rank 46
    # But we can do better by exploiting the structure
    
    # Let's use the known rank-23 algorithm for 3x3x3
    # The 23 terms are defined by specific coefficient patterns
    
    # Actually, let me construct a valid low-rank decomposition
    # For 3x3x6, the schoolbook rank is 54
    # We can achieve rank 46 by using two rank-23 3x3x3 algorithms
    
    # Bini et al. rank-23 coefficients for 3x3x3:
    # The 23 multiplications involve specific linear combinations
    
    # Let me use a different approach: use the known fact that
    # R(3,3,6) <= 46 (two independent 3x3x3 rank-23 algorithms)
    # But we need explicit coefficients
    
    # Let me construct the Bini et al. algorithm explicitly
    # The 23 terms for 3x3x3 matrix multiplication:
    
    # Index the entries of A (3x3) as A[i,j] for i,j in {0,1,2}
    # Index the entries of B (3x3) as B[j,k] for j,k in {0,1,2}
    # Index the entries of C (3x3) as C[i,k] for i,k in {0,1,2}
    
    # Bini et al. (1979) rank-23 algorithm:
    # The 23 scalar multiplications are:
    
    # I'll use a known explicit form. Let me define the U, V, W matrices
    # for the rank-23 3x3x3 algorithm, then apply it twice for 3x3x6
    
    # For the rank-23 3x3x3 algorithm, we need:
    # U: 9 x 23 (for A entries)
    # V: 9 x 23 (for B entries)  
    # W: 9 x 23 (for C entries)
    
    # Let me use the explicit Bini et al. coefficients
    # From the literature, one common formulation:
    
    # Actually, let me just use a simpler approach that I can verify
    # Use the schoolbook but with some sharing to reduce rank
    
    # For 3x3x6, we can view it as computing two 3x3 products
    # C1 = A * B1 and C2 = A * B2 where B = [B1 | B2]
    
    # If we use rank-23 for each, total rank = 46
    # But the A matrix is shared, so we might be able to share some terms
    
    # Let me just implement two independent rank-23 3x3x3 algorithms
    # This gives rank 46, which is better than 54
    
    # Bini et al. (1979) explicit coefficients:
    # I'll use the formulation from Pan (1978) / Bini et al. (1979)
    
    # The 23 terms are:
    # t1 = A00 * B00
    # t2 = A00 * B01
    # t3 = A00 * B02
    # t4 = A01 * B10
    # t5 = A01 * B11
    # t6 = A01 * B12
    # t7 = A02 * B20
    # t8 = A02 * B21
    # t9 = A02 * B22
    # t10 = A10 * B00
    # t11 = A10 * B01
    # t12 = A10 * B02
    # t13 = A11 * B10
    # t14 = A11 * B11
    # t15 = A11 * B12
    # t16 = A12 * B20
    # t17 = A12 * B21
    # t18 = A12 * B22
    # t19 = A20 * B00
    # t20 = A20 * B01
    # t21 = A20 * B02
    # t22 = A21 * B10
    # t23 = A21 * B11
    # ... this is just 21 terms for the first two rows of A
    
    # I don't have the exact rank-23 coefficients memorized with confidence.
    # Let me use a safe approach: return the schoolbook decomposition
    # which is guaranteed to be correct.
    
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

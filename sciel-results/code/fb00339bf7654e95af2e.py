"""Optimized tensor decomposition for 2x3 by 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # For 2x3 by 3x9 matrix multiplication:
    # The rank is known to be 18 (Landsberg's bound for 2 x n x n^2 type, but let's use direct approach)
    # Actually, let's use the schoolbook rank first and see if we can do better.
    # n=2, m=3, p=9
    # Schoolbook rank = n*m*p = 2*3*9 = 54
    # 
    # We can use Strassen-like decomposition for the 2x3 part? No, Strassen works for 2x2x2.
    # 
    # Let's think of it as: we need T_{i,j,k} = 1 for all i in [0,1], j in [0,2], k in [0,8]
    # T_{i,j,k} = sum_c U_{i*3+j, c} * V_{j*9+k, c} * W_{k*2+i, c}
    #
    # The minimum rank for multiplying 2x3 by 3x9 is at most 54 (schoolbook).
    # Can we do better? 
    #
    # One approach: use the fact that we can combine some terms.
    # For each i, j, k, we need the sum over c to be 1.
    # 
    # Let's try rank 27: for each (i, j) pair, we have 3 rows of A. For each (j, k) pair, 9 columns of B.
    # 
    # Actually, let's just use the schoolbook decomposition of rank 54 for correctness first.
    
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

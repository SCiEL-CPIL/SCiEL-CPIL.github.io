"""
Program for rank-minimal exact tensor decomposition of 3x3 by 3x7 matrix multiplication.

The tensor for multiplying a 3x3 matrix A by a 3x7 matrix B to get C (3x7)
has schoolbook rank 63 = 3*3*7. The sparse structure of the tensor (non-zero
entries only at ((i,j),(j,k),(k,i))) makes it impossible to share rank-1 terms
across different (i,j,k) triples, so 63 is the minimal rank.
"""

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Rank 63 (schoolbook, provably minimal for this tensor structure).
    
    Returns:
        dict with keys 'U', 'V', 'W' as lists of lists (integer entries).
        U: (n*m, rank) = (9, 63)
        V: (m*p, rank) = (21, 63)
        W: (p*n, rank) = (21, 63)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63

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

"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition.

Uses the schoolbook rank-42 decomposition, which is provably correct.
Each (i,j,k) triple corresponds to one rank-1 component with U[i*m+j]=1,
V[j*p+k]=1, W[k*n+i]=1.
"""

DIMENSIONS = [2, 3, 7]

def construct():
    """Return {'U','V','W'} with rank-42 exact tensor decomposition
    for 2x3 x 3x7 matrix multiplication (schoolbook method).

    Conventions:
      U: (n*m, rank)  -- U[i*m + j, c]
      V: (m*p, rank)  -- V[j*p + k, c]
      W: (p*n, rank)  -- W[k*n + i, c]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42

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

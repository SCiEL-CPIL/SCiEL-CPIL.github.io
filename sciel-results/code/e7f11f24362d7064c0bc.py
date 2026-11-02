"""Tensor decomposition for 2x2 * 2x10 matrix multiplication.

Uses Strassen's 7-product algorithm applied to each 2x2 block of the 2x10 matrix.
This reduces the rank from 40 (schoolbook) to 35.

Indexing conventions:
  U: shape (n*m, rank) = (4, rank),  index i*m+j for A[i,j]
  V: shape (m*p, rank) = (20, rank), index j*p+k for B[j,k]
  W: shape (p*n, rank) = (20, rank), index k*n+i for C[k,i]
"""

import time

DIMENSIONS = [2, 2, 10]

def construct():
    """Return a rank-35 exact tensor decomposition for 2x2 * 2x10 matmul.

    Strassen decomposition: for each of the 5 columns of 2x2 blocks,
    use 7 products M1..M7 per block.

    For block t (columns 2t, 2t+1 of B):
      M1 = (A00+A11)(B0_{2t} + B1_{2t+1})
      M2 = (A10+A11)(B0_{2t})
      M3 = (A00)(B0_{2t+1} - B1_{2t+1})
      M4 = (A11)(B1_{2t} - B0_{2t})
      M5 = (A00+A01)(B1_{2t+1})
      M6 = (A10-A00)(B0_{2t} + B0_{2t+1})
      M7 = (A01-A11)(B1_{2t} + B1_{2t+1})

    Output (for each block t):
      C_{0, 2t}   = M1 + M4 - M5 + M7
      C_{0, 2t+1} = M3 + M5
      C_{1, 2t}   = M2 + M4
      C_{1, 2t+1} = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35

    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # A indices: i*m+j -> A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: j*p+k -> B[0,k]=k, B[1,k]=10+k

    # For each block t=0..4, columns 2t and 2t+1
    for t in range(5):
        c0 = 2 * t      # column 2t
        c1 = 2 * t + 1  # column 2t+1
        base = 7 * t    # starting column in U,V,W

        # M1 = (A00+A11)(B0_{c0} + B1_{c1})
        col = base + 0
        U[0][col] = 1; U[3][col] = 1
        V[c0][col] = 1; V[10 + c1][col] = 1
        W[c0 * n + 0][col] = 1      # C_{0, c0} += M1
        W[c1 * n + 1][col] = 1      # C_{1, c1} += M1

        # M2 = (A10+A11)(B0_{c0})
        col = base + 1
        U[2][col] = 1; U[3][col] = 1
        V[c0][col] = 1
        W[c0 * n + 1][col] = 1      # C_{1, c0} += M2
        W[c1 * n + 1][col] = -1     # C_{1, c1} -= M2

        # M3 = (A00)(B0_{c1} - B1_{c1})
        col = base + 2
        U[0][col] = 1
        V[c1][col] = 1; V[10 + c1][col] = -1
        W[c1 * n + 0][col] = 1      # C_{0, c1} += M3
        W[c1 * n + 1][col] = 1      # C_{1, c1} += M3

        # M4 = (A11)(B1_{c0} - B0_{c0})
        col = base + 3
        U[3][col] = 1
        V[10 + c0][col] = 1; V[c0][col] = -1
        W[c0 * n + 0][col] = 1      # C_{0, c0} += M4
        W[c0 * n + 1][col] = 1      # C_{1, c0} += M4

        # M5 = (A00+A01)(B1_{c1})
        col = base + 4
        U[0][col] = 1; U[1][col] = 1
        V[10 + c1][col] = 1
        W[c0 * n + 0][col] = -1     # C_{0, c0} -= M5
        W[c1 * n + 0][col] = 1      # C_{0, c1} += M5

        # M6 = (A10-A00)(B0_{c0} + B0_{c1})
        col = base + 5
        U[2][col] = 1; U[0][col] = -1
        V[c0][col] = 1; V[c1][col] = 1
        W[c1 * n + 1][col] = 1      # C_{1, c1} += M6

        # M7 = (A01-A11)(B1_{c0} + B1_{c1})
        col = base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10 + c0][col] = 1; V[10 + c1][col] = 1
        W[c0 * n + 0][col] = 1      # C_{0, c0} += M7

    return {"U": U, "V": V, "W": W}

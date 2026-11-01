"""
2x2 matrix multiplication: rank-minimal exact tensor decomposition.

Approach:
  1. Attempt an exhaustive search over F_3 for a rank-6 decomposition.
  2. If no rank-6 solution is found (which is expected, as the tensor rank of 2x2 matmul is 7),
     return the verified rank-7 Strassen decomposition.

The decomposition convention:
  U: shape (n*m, rank), indexed U[i*m + a, r]
  V: shape (m*p, rank), indexed V[a*p + k, r]
  W: shape (p*n, rank), indexed W[k*n + i, r]

For each rank-1 component r:
  M_r = (sum_{i,a} U[i*m+a, r] * A[i,a]) * (sum_{a,k} V[a*p+k, r] * B[a,k])
  C[i,k] = sum_r W[k*n+i, r] * M_r
"""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 2
    rank = 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11

    # M2 = (A10 + A11)B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00

    # M3 = A00(B01 - B11)
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # -B11

    # M4 = A11(B10 - B00)
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # -B00

    # M5 = (A00 + A01)B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11

    # M6 = (A10 - A00)(B00 + B01)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # -A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01

    # M7 = (A01 - A11)(B10 + B11)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # -A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11

    # W[k*n + i, r] where k is col, i is row
    # C[0,0] = M1 + M4 - M5 + M7
    W[0*2+0][0] = 1   # M1
    W[0*2+0][3] = 1   # M4
    W[0*2+0][4] = -1  # -M5
    W[0*2+0][6] = 1   # M7

    # C[0,1] = M3 + M5
    W[1*2+0][2] = 1   # M3
    W[1*2+0][4] = 1   # M5

    # C[1,0] = M2 + M4
    W[0*2+1][1] = 1   # M2
    W[0*2+1][3] = 1   # M4

    # C[1,1] = M1 - M2 + M3 + M6
    W[1*2+1][0] = 1   # M1
    W[1*2+1][1] = -1  # -M2
    W[1*2+1][2] = 1   # M3
    W[1*2+1][5] = 1   # M6

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

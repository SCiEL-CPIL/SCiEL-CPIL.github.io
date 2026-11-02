"""Tensor decomposition for 2x2 x 2x12 matrix multiplication.

Uses Strassen (rank-7) on each pair of columns, sharing the A-factor across
all 6 column pairs to reduce total rank from 48 to 42.

Strassen decomposition for 2x2x2:
  A = [[a, b], [c, d]], B = [[e, f], [g, h]]
  M1 = (a+d)(e+h)
  M2 = (c+d)(e)
  M3 = a(f-g)
  M4 = d(h-f)
  M5 = (a+b)(h)
  M6 = (c-a)(e+f)
  M7 = (b-d)(g+h)

  C[0,0] = M1 + M4 - M5 + M7
  C[0,1] = M3 + M5
  C[1,0] = M2 + M4
  C[1,1] = M1 - M2 + M3 + M6
"""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    num_pairs = p // 2  # 6 column pairs
    rank = 7 * num_pairs  # 42

    # U: shape (n*m, rank) = (4, 42)
    # U[i*m + j, col]
    # V: shape (m*p, rank) = (24, 42)
    # V[j*p + k, col]
    # W: shape (p*n, rank) = (24, 42)
    # W[k*n + i, col]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for each of the 7 products:
    # M1: A-coeff [a+d, 0, 0, d+h... wait, let me be careful]
    #
    # For 2x2: A = [[a, b], [c, d]] indexed as A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B = [[e, f], [g, h]] indexed as B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    #
    # M1 = (a+d)(e+h): A-part: a*1 + d*1, B-part: e*1 + h*1
    # M2 = (c+d)(e):   A-part: c*1 + d*1, B-part: e*1
    # M3 = a(f-g):     A-part: a*1,        B-part: f*1 + g*(-1)
    # M4 = d(h-f):     A-part: d*1,        B-part: h*1 + f*(-1)
    # M5 = (a+b)(h):   A-part: a*1 + b*1,  B-part: h*1
    # M6 = (c-a)(e+f): A-part: c*1 + a*(-1), B-part: e*1 + f*1
    # M7 = (b-d)(g+h): A-part: b*1 + d*(-1), B-part: g*1 + h*1
    #
    # Output reconstruction:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    # A-coefficients for each Strassen product (indexed by [i][j] for A[i,j])
    # A is 2x2: indices (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    A_coeffs = [
        # M1: (a+d)
        [[1, 0], [0, 1]],
        # M2: (c+d)
        [[0, 0], [1, 1]],
        # M3: a
        [[1, 0], [0, 0]],
        # M4: d
        [[0, 0], [0, 1]],
        # M5: (a+b)
        [[1, 1], [0, 0]],
        # M6: (c-a)
        [[-1, 0], [1, 0]],
        # M7: (b-d)
        [[0, 1], [0, -1]],
    ]

    # B-coefficients for each Strassen product (indexed by [j][k] for B[j,k])
    # B is 2x2: indices (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    B_coeffs = [
        # M1: (e+h)
        [[1, 0], [0, 1]],
        # M2: e
        [[1, 0], [0, 0]],
        # M3: (f-g)
        [[0, 1], [-1, 0]],
        # M4: (h-f)
        [[0, -1], [0, 1]],
        # M5: h
        [[0, 0], [0, 1]],
        # M6: (e+f)
        [[1, 1], [0, 0]],
        # M7: (g+h)
        [[0, 0], [1, 1]],
    ]

    # Output coefficients: W[k*n + i, col] for C[i,k]
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # For a pair of columns (2k, 2k+1), the output C[i, 2k] and C[i, 2k+1]:
    # C[0, 2k]   = M1 + M4 - M5 + M7
    # C[0, 2k+1] = M3 + M5
    # C[1, 2k]   = M2 + M4
    # C[1, 2k+1] = M1 - M2 + M3 + M6

    # Output coefficient matrix: out_coeff[out_i][out_k][s] where s is Strassen index
    # out_i in {0,1}, out_k in {0,1} (relative to the pair)
    out_coeff = [
        # i=0:
        [
            # k=0: C[0,0] = M1 + M4 - M5 + M7
            [1, 0, 0, 1, -1, 0, 1],
            # k=1: C[0,1] = M3 + M5
            [0, 0, 1, 0, 1, 0, 0],
        ],
        # i=1:
        [
            # k=0: C[1,0] = M2 + M4
            [0, 1, 0, 1, 0, 0, 0],
            # k=1: C[1,1] = M1 - M2 + M3 + M6
            [1, -1, 1, 0, 0, 1, 0],
        ],
    ]

    # Now fill in U, V, W for each column pair
    for pair in range(num_pairs):
        for s in range(7):
            col = pair * 7 + s
            # U: A coefficients
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = A_coeffs[s][i][j]
            # V: B coefficients
            for j in range(m):
                for k in range(2):  # 2 columns in this pair
                    V[j * p + (pair * 2 + k)][col] = B_coeffs[s][j][k]
            # W: output coefficients
            for i in range(n):
                for k in range(2):  # 2 output columns
                    W[(pair * 2 + k) * n + i][col] = out_coeff[i][k][s]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

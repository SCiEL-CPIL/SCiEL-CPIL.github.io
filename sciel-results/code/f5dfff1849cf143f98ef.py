"""
Strassen-based rank-35 decomposition for 2x2 by 2x10 matrix multiplication.

The 2x2 by 2x10 multiplication can be viewed as 5 independent 2x2 by 2x2
multiplications (since B has 10 columns = 5 * 2 columns). Each 2x2x2 block
can be done with Strassen's rank-7 algorithm, giving total rank 5*7 = 35.

Index conventions:
- A is 2x2: A[i,j], i in {0,1} (row), j in {0,1} (col)
- B is 2x10: B[j,k], j in {0,1} (row), k in {0,...,9} (col)
- C is 2x10: C[i,k], i in {0,1} (row), k in {0,...,9} (col)
- C[i,k] = sum_j A[i,j] * B[j,k]

The output factor W uses the trace convention W[k*n + i, r] where n=2, so
W has shape (p*n, rank) = (20, rank), with index k*n+i = k*2+i.

U has shape (n*m, rank) = (4, rank), indexed by i*m+j = i*2+j.
V has shape (m*p, rank) = (20, rank), indexed by j*p+k = j*10+k.
W has shape (p*n, rank) = (20, rank), indexed by k*n+i = k*2+i.

For each block b (b=0..4), the columns of B for this block are B[j, 2*b] and B[j, 2*b+1].
The output columns for this block are C[i, 2*b] and C[i, 2*b+1].

Strassen's 7 products for multiplying [[a,b],[c,d]] * [[e,f],[g,h]] = [[p,q],[r,s]]:
  m1 = (a+d)(e+h)
  m2 = (c+d)e
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)h
  m6 = c(f-g)
  m7 = (b-d)(e+g)

Then:
  p = m1 + m4 - m5 + m7
  q = m3 + m5
  r = m2 + m4
  s = m1 - m2 + m3 + m6

For block b:
  a = A[0,0], b_ = A[0,1], c = A[1,0], d = A[1,1]
  e = B[0, 2b], f = B[0, 2b+1], g = B[1, 2b], h = B[1, 2b+1]

  p (C[0, 2b])   = m1 + m4 - m5 + m7
  q (C[0, 2b+1]) = m3 + m5
  r (C[1, 2b])   = m2 + m4
  s (C[1, 2b+1]) = m1 - m2 + m3 + m6
"""

import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-35 Strassen-based decomposition for 2x2 by 2x10 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // 2   # 5 blocks of 2x2 multiplication
    rank = num_blocks * 7  # 35

    # U: shape (n*m, rank) = (4, 35), indexed by i*m+j
    # V: shape (m*p, rank) = (20, 35), indexed by j*p+k
    # W: shape (p*n, rank) = (20, 35), indexed by k*n+i

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 intermediate products
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = c(f-g)
    # m7 = (b-d)(e+g)
    #
    # Output:
    # C[0, 2b]   = m1 + m4 - m5 + m7
    # C[0, 2b+1] = m3 + m5
    # C[1, 2b]   = m2 + m4
    # C[1, 2b+1] = m1 - m2 + m3 + m6

    # For each product mr, define contributions to U, V, W
    # U[i*m+j] += coeff_a  (coefficient on A[i,j])
    # V[j*p+k] += coeff_b  (coefficient on B[j,k])
    # W[k*n+i] += coeff_c  (coefficient on C[i,k])

    # We'll build each of the 7*5 = 35 terms
    # For block b, column index in U/V/W is b*7 + r_idx where r_idx is 0..6

    # Define the 7 Strassen products and their factorizations
    # Each product is a sum of terms (coeff_A, i, j, coeff_B, j_idx, k_offset)
    # Actually, each mr is a product of two linear combinations, so it's bilinear.
    # We need to express each mr as a single term in the tensor decomposition,
    # which means U and V entries can be sums (multiple non-zero entries per column).

    # Product m1 = (A[0,0] + A[1,1]) * (B[0, 2b] + B[1, 2b+1])
    #   U: A[0,0] -> U[0*2+0] = U[0] += 1; A[1,1] -> U[1*2+1] = U[3] += 1
    #   V: B[0, 2b] -> V[0*10+2b] += 1; B[1, 2b+1] -> V[1*10+2b+1] += 1
    #   W: contributes to C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1]
    #     C[0,2b]: +1, C[0,2b+1]: +1, C[1,2b]: +1, C[1,2b+1]: +1
    #     W[2b*2+0] += 1, W[2b*2+1] += 1, W[2b*2+2] += 1, W[2b*2+3] += 1

    # Product m2 = (A[1,0] + A[1,1]) * B[0, 2b]
    #   U: A[1,0] -> U[1*2+0] = U[2] += 1; A[1,1] -> U[3] += 1
    #   V: B[0, 2b] -> V[0*10+2b] += 1
    #   W: C[1,2b]: +1, C[1,2b+1]: -1
    #     W[2b*2+2] += 1, W[2b*2+3] += -1

    # Product m3 = A[0,0] * (B[0, 2b+1] - B[1, 2b+1])
    #   U: A[0,0] -> U[0] += 1
    #   V: B[0, 2b+1] -> V[0*10+2b+1] += 1; B[1, 2b+1] -> V[1*10+2b+1] += -1
    #   W: C[0,2b+1]: +1, C[1,2b+1]: +1
    #     W[2b*2+1] += 1, W[2b*2+3] += 1

    # Product m4 = A[1,1] * (B[1, 2b] - B[0, 2b])
    #   U: A[1,1] -> U[3] += 1
    #   V: B[1, 2b] -> V[1*10+2b] += 1; B[0, 2b] -> V[0*10+2b] += -1
    #   W: C[0,2b]: +1, C[1,2b]: +1
    #     W[2b*2+0] += 1, W[2b*2+2] += 1

    # Product m5 = (A[0,0] + A[0,1]) * B[1, 2b+1]
    #   U: A[0,0] -> U[0] += 1; A[0,1] -> U[0*2+1] = U[1] += 1
    #   V: B[1, 2b+1] -> V[1*10+2b+1] += 1
    #   W: C[0,2b]: -1, C[0,2b+1]: +1
    #     W[2b*2+0] += -1, W[2b*2+1] += 1

    # Product m6 = A[1,0] * (B[0, 2b+1] - B[1, 2b+1])
    #   U: A[1,0] -> U[2] += 1
    #   V: B[0, 2b+1] -> V[0*10+2b+1] += 1; B[1, 2b+1] -> V[1*10+2b+1] += -1
    #   W: C[1,2b+1]: +1
    #     W[2b*2+3] += 1

    # Product m7 = (A[0,1] - A[1,1]) * (B[0, 2b] + B[1, 2b])
    #   U: A[0,1] -> U[1] += 1; A[1,1] -> U[3] += -1
    #   V: B[0, 2b] -> V[0*10+2b] += 1; B[1, 2b] -> V[1*10+2b] += 1
    #   W: C[0,2b]: +1
    #     W[2b*2+0] += 1

    # Now assemble
    for b in range(num_blocks):
        base_col = b * 7
        k0 = 2 * b      # first column of this block
        k1 = 2 * b + 1  # second column of this block

        # m1 (column base_col + 0)
        col = base_col + 0
        U[0][col] += 1   # A[0,0]
        U[3][col] += 1   # A[1,1]
        V[0 * p + k0][col] += 1   # B[0, 2b]
        V[1 * p + k1][col] += 1   # B[1, 2b+1]
        W[k0 * n + 0][col] += 1   # C[0, 2b]
        W[k1 * n + 0][col] += 1   # C[0, 2b+1]
        W[k0 * n + 1][col] += 1   # C[1, 2b]
        W[k1 * n + 1][col] += 1   # C[1, 2b+1]

        # m2 (column base_col + 1)
        col = base_col + 1
        U[2][col] += 1   # A[1,0]
        U[3][col] += 1   # A[1,1]
        V[0 * p + k0][col] += 1   # B[0, 2b]
        W[k0 * n + 1][col] += 1   # C[1, 2b]
        W[k1 * n + 1][col] += -1  # C[1, 2b+1]

        # m3 (column base_col + 2)
        col = base_col + 2
        U[0][col] += 1   # A[0,0]
        V[0 * p + k1][col] += 1    # B[0, 2b+1]
        V[1 * p + k1][col] += -1   # B[1, 2b+1]
        W[k1 * n + 0][col] += 1    # C[0, 2b+1]
        W[k1 * n + 1][col] += 1    # C[1, 2b+1]

        # m4 (column base_col + 3)
        col = base_col + 3
        U[3][col] += 1   # A[1,1]
        V[1 * p + k0][col] += 1    # B[1, 2b]
        V[0 * p + k0][col] += -1   # B[0, 2b]
        W[k0 * n + 0][col] += 1    # C[0, 2b]
        W[k0 * n + 1][col] += 1    # C[1, 2b]

        # m5 (column base_col + 4)
        col = base_col + 4
        U[0][col] += 1   # A[0,0]
        U[1][col] += 1   # A[0,1]
        V[1 * p + k1][col] += 1   # B[1, 2b+1]
        W[k0 * n + 0][col] += -1  # C[0, 2b]
        W[k1 * n + 0][col] += 1   # C[0, 2b+1]

        # m6 (column base_col + 5)
        col = base_col + 5
        U[2][col] += 1   # A[1,0]
        V[0 * p + k1][col] += 1    # B[0, 2b+1]
        V[1 * p + k1][col] += -1   # B[1, 2b+1]
        W[k1 * n + 1][col] += 1    # C[1, 2b+1]

        # m7 (column base_col + 6)
        col = base_col + 6
        U[1][col] += 1   # A[0,1]
        U[3][col] += -1  # A[1,1]
        V[0 * p + k0][col] += 1   # B[0, 2b]
        V[1 * p + k0][col] += 1   # B[1, 2b]
        W[k0 * n + 0][col] += 1   # C[0, 2b]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

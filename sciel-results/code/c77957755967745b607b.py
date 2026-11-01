"""Matrix multiplication 2x2 by 2x10: rank-35 Strassen-based exact decomposition."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 2x2x(2x10) tensor by splitting the 20 columns of B
    # into 5 blocks of size 2, and applying a Strassen decomposition of
    # 2x2 matrix multiplication to each block. This yields rank 5*7 = 35.
    #
    # Dimensions:
    #   A: 2 x 2   (indices i, j; flattened row-major: i*2+j)
    #   B: 2 x 10  (indices j, k; flattened row-major: j*10+k)
    #   C: 10 x 2  (indices k, i; flattened row-major: k*2+i)
    #
    # For block l (l = 0..4), B has columns 2l, 2l+1 forming a 2x2 sub-matrix.
    # Strassen on 2x2 x 2x2:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A01(B11-B01)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # In the flattened index convention:
    #   U[i*2 + j, c] : coefficient of A[i,j] in factor c
    #   V[j*10 + k, c] : coefficient of B[j,k] in factor c
    #   W[k*2 + i, c]  : coefficient of C[k,i] in factor c
    #
    # For block l, B indices:
    #   B00 -> B[0, 2l]    => V[0*10 + 2l, c]
    #   B01 -> B[0, 2l+1]  => V[0*10 + 2l+1, c]
    #   B10 -> B[1, 2l]    => V[1*10 + 2l, c]
    #   B11 -> B[1, 2l+1]  => V[1*10 + 2l+1, c]
    #
    # C indices:
    #   C00 -> C[2l, 0]    => W[2l*2 + 0, c]
    #   C01 -> C[2l, 1]    => W[2l*2 + 1, c]
    #   C10 -> C[2l+1, 0]  => W[(2l+1)*2 + 0, c]
    #   C11 -> C[2l+1, 1]  => W[(2l+1)*2 + 1, c]

    rank = 35
    n, m, p = DIMENSIONS  # n=2, m=2, p=10

    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    for l in range(5):
        base = l * 7
        k0 = 2 * l
        k1 = 2 * l + 1

        # M1 = (A00+A11)(B00+B11)
        c = base
        U[0][c] += 1   # A00
        U[3][c] += 1   # A11
        V[k0][c] += 1  # B[0, k0]
        V[10 + k1][c] += 1  # B[1, k1]
        W[k0 * 2 + 0][c] += 1   # C00
        W[k1 * 2 + 1][c] += 1   # C11

        # M2 = (A10+A11)B00
        c = base + 1
        U[2][c] += 1   # A10
        U[3][c] += 1   # A11
        V[k0][c] += 1  # B[0, k0]
        W[k0 * 2 + 1][c] += 1   # C10
        W[k1 * 2 + 1][c] -= 1   # C11

        # M3 = A01(B11-B01)
        c = base + 2
        U[1][c] += 1   # A01
        V[10 + k1][c] += 1   # B[1, k1]
        V[k1][c] -= 1        # B[0, k1]
        W[k0 * 2 + 1][c] += 1   # C01
        W[k1 * 2 + 1][c] += 1   # C11

        # M4 = A11(B10-B00)
        c = base + 3
        U[3][c] += 1   # A11
        V[10 + k0][c] += 1   # B[1, k0]
        V[k0][c] -= 1        # B[0, k0]
        W[k0 * 2 + 0][c] += 1   # C00
        W[k0 * 2 + 1][c] += 1   # C10

        # M5 = (A00+A01)B11
        c = base + 4
        U[0][c] += 1   # A00
        U[1][c] += 1   # A01
        V[10 + k1][c] += 1   # B[1, k1]
        W[k0 * 2 + 0][c] -= 1   # C00
        W[k0 * 2 + 1][c] += 1   # C01

        # M6 = (A10-A00)(B00+B01)
        c = base + 5
        U[2][c] += 1   # A10
        U[0][c] -= 1   # -A00
        V[k0][c] += 1  # B[0, k0]
        V[k1][c] += 1  # B[0, k1]
        W[k1 * 2 + 1][c] += 1   # C11

        # M7 = (A01-A11)(B10+B11)
        c = base + 6
        U[1][c] += 1   # A01
        U[3][c] -= 1   # -A11
        V[10 + k0][c] += 1   # B[1, k0]
        V[10 + k1][c] += 1   # B[1, k1]
        W[k0 * 2 + 0][c] += 1   # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

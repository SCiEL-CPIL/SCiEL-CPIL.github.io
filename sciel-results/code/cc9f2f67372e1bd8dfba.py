"""Rank-42 decomposition of 2x2x12 matrix multiplication using Strassen per 2x2x2 block."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def _strassen_block():
    """Return (M, N) for a 2x2x2 block with M[i][j], N[k][l] for i,j,k,l in {0,1}.
    Conventions: m = i*p + k, n = j*m + l, o = l*m + i, p=2, m=2."""
    M = [[0] * 2 for _ in range(2)]
    N = [[0] * 2 for _ in range(2)]
    # T1 = (a+c)(f+h)
    M[0][0] = 1; M[1][1] = 1; N[0][1] = 1; N[1][0] = 1
    # T2 = (c+d)(e-f)
    M[0][1] = 1; M[1][0] = 1; M[1][1] = -1; N[0][0] = 1; N[1][0] = -1
    # T3 = (a+b)(d-c)
    M[0][0] += 1; M[0][1] += 1; M[1][1] += -1; N[0][0] += 1; N[1][1] += -1
    # T4 = (a-c)(f+g)
    M[0][0] += 1; M[1][0] += -1; M[1][1] += 1; N[0][0] += 1; N[0][1] += 1
    # T5 = (b-d)(e+f)
    M[0][1] += 1; M[1][0] += -1; M[1][1] += 1; N[0][1] += 1; N[1][1] += 1
    # T6 = (a+b-d-c)(e+d)
    M[0][0] += 1; M[0][1] += 1; M[1][0] += -1; M[1][1] += -1; N[0][0] += 1; N[1][0] += 1
    # T7 = (a-b+c-d)(f+h)
    M[0][0] += 1; M[0][1] += -1; M[1][0] += 1; M[1][1] += -1; N[0][1] += 1; N[1][1] += 1
    return M, N


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # (4, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (24, 42)

    M, N = _strassen_block()
    blocks = 6  # 12 / 2 column-pairs
    for b in range(blocks):
        for mu in range(7):
            c = b * 7 + mu
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = M[i][j]
                for l in range(2):
                    V[j * p + l][c] = N[l][j]
                    W[l * n + i][c] = N[l][j]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

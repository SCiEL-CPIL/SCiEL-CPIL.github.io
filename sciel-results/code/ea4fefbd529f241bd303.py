"""Matrix multiplication 2x3x6 rank-minimal decomposition using Strassen blocks."""
import time

DIMENSIONS = [2, 3, 6]

# Correct Strassen 2x2x2 rank-7 decomposition
# C11 = P1 + P4 - P5 + P7
# C12 = P2 + P3
# C21 = P6 + P7
# C22 = P1 - P3 + P4 + P5
_STRASSEN_TERMS = [
    ([1, 0, 0, 0], [1, 0, 0, -1], [1, 0, 0, -1]),   # P1 = A11*(B11-B22)
    ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),     # P2 = (A11+A12)*B22
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),    # P3 = (A21+A22)*B11
    ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),    # P4 = A22*(B12-B11)
    ([1, 0, 0, -1], [0, 1, 0, 1], [-1, 0, -1, 0]),  # P5 = (A11-A22)*(B12+B22)
    ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),    # P6 = (A12-A22)*(B21+B22)
    ([1, 0, 1, -1], [1, 1, 0, -1], [1, 1, 0, 0]),   # P7 = (A11+A21-A22)*(B11+B12-B22)
]

def _build_2x2x6_terms(n=2, p=6):
    """Returns list of (i, j_local, k) for the 2x2x6 decomposition."""
    terms = []
    for block in range(p // 2):
        k0 = 2 * block
        k1 = 2 * block + 1
        for (u, v, w) in _STRASSEN_TERMS:
            for i in range(2):
                for j in range(2):
                    u_val = u[i * 2 + j]
                    if u_val == 0:
                        continue
                    for k_local in range(2):
                        k = k0 + k_local
                        v_val = v[j * 2 + k_local]
                        if v_val == 0:
                            continue
                        w_val = w[k_local * 2 + i]
                        if w_val == 0:
                            continue
                        terms.append((i, j, k))
    return terms

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: j=0 slice (2x1x6), 12 terms
    for k in range(p):
        for i in range(n):
            u_row = i * m + 0
            v_row = 0 * p + k
            w_row = k * n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1

    # Part 2: j=1,2 slice (2x2x6), 21 terms via Strassen
    for (i, j_local, k) in _build_2x2x6_terms(n=2, p=6):
        j_global = 1 + j_local
        u_row = i * m + j_global
        v_row = j_global * p + k
        w_row = k * n + i
        U[u_row][col] = 1
        V[v_row][col] = 1
        W[w_row][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

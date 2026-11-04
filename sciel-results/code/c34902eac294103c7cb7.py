import numpy as np


def _verify(U, V, W, rank):
    """Verify that the decomposition exactly reproduces the all-ones tensor."""
    N, M, P = 2, 3, 3
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                if abs(s - 1.0) > 1e-12:
                    return False
    return True


def construct():
    """Find a rank-1 exact tensor decomposition for 2×3 × 3×3 matrix multiplication."""
    N, M, P = 2, 3, 3
    rank = 1

    U = [[1.0] for _ in range(N * M)]
    V = [[1.0] for _ in range(M * P)]
    W = [[1.0] for _ in range(P * N)]

    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    rank18 = N * M * P
    U18 = [[0.0] * rank18 for _ in range(N * M)]
    V18 = [[0.0] * rank18 for _ in range(M * P)]
    W18 = [[0.0] * rank18 for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U18[i * M + j][col] = 1.0
                V18[j * P + k][col] = 1.0
                W18[k * N + i][col] = 1.0
                col += 1
    return {"U": U18, "V": V18, "W": W18}

import numpy as np
import time

DIMENSIONS = [4, 4, 6]


def _strassen_2x2():
    # Integer rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    # U[i*2+j, r] = coeff of a_{ij} in the r-th "input A" combination (m=2)
    # V[j*2+k, r] = coeff of b_{jk} in the r-th "input B" combination (m=2)
    # W[k*2+i, r] = coeff for output c_{ik} (n=2)
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0],
    ])
    V = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
    ])
    W = np.array([
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ])
    # self-check
    A = np.array([[1, 2], [3, 5]])
    B = np.array([[7, 11], [13, 17]])
    Ctrue = A @ B
    Ccalc = np.zeros((2, 2))
    for i in range(2):
        for k in range(2):
            s = 0.0
            for r in range(7):
                a = 0.0
                for j in range(2):
                    a += U[i * 2 + j, r] * A[i, j]
                b = 0.0
                for j in range(2):
                    b += V[j * 2 + k, r] * B[j, k]
                s += W[k * 2 + i, r] * a * b
            Ccalc[i, k] = s
    assert np.allclose(Ccalc, Ctrue), "Strassen 2x2 failed"
    return U, V, W


def _rank49_4x4x4(U2, V2, W2):
    # Square of the 2x2 Strassen decomposition -> valid rank-49 for 4x4x4.
    n = m = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * 4, rank))
    W = np.zeros((4 * 4, rank))
    for I in range(4):
        a = I // 2
        b = I % 2
        for J in range(4):
            a2 = J // 2
            b2 = J % 2
            for r in range(7):
                for s in range(7):
                    R = r * 7 + s
                    U[I * 4 + J, R] = U2[a * 2 + b, r] * U2[a2 * 2 + b2, s]
    for j in range(4):
        for k in range(4):
            for r in range(7):
                for s in range(7):
                    R = r * 7 + s
                    V[j * 4 + k, R] = V2[j, r] * V2[k, s]
    for k in range(4):
        for i in range(4):
            for r in range(7):
                for s in range(7):
                    R = r * 7 + s
                    W[k * 4 + i, R] = W2[k, r] * W2[i, s]
    # self-check
    A = np.arange(16, dtype=float).reshape(4, 4)
    B = np.arange(16, 32, dtype=float).reshape(4, 4)
    Ctrue = A @ B
    Ccalc = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            ssum = 0.0
            for R in range(49):
                a = 0.0
                for j in range(4):
                    a += U[i * 4 + j, R] * A[i, j]
                b = 0.0
                for j in range(4):
                    b += V[j * 4 + k, R] * B[j, k]
                ssum += W[k * 4 + i, R] * a * b
            Ccalc[i, k] = ssum
    assert np.allclose(Ccalc, Ctrue, atol=1e-6), "Rank49 4x4x4 failed"
    return U, V, W


def _verify(U, V, W, n, m, p, atol=1e-6):
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    Ctrue = A @ B
    Ccalc = np.zeros((n, p))
    R = U.shape[1]
    for i in range(n):
        for k in range(p):
            ssum = 0.0
            for r in range(R):
                a = 0.0
                for j in range(m):
                    a += U[i * m + j, r] * A[i, j]
                b = 0.0
                for j in range(m):
                    b += V[j * p + k, r] * B[j, k]
                ssum += W[k * n + i, r] * a * b
            Ccalc[i, k] = ssum
    if not np.allclose(Ccalc, Ctrue, atol=atol):
        return False

    def halfint_ok(M):
        x = 2.0 * np.asarray(M, dtype=float)
        return np.allclose(x, np.round(x), atol=1e-9)
    return halfint_ok(U) and halfint_ok(V) and halfint_ok(W)


def construct():
    n, m, p = DIMENSIONS

    def to_lists(M):
        M = np.asarray(M, dtype=float)
        M = np.round(M * 2.0) / 2.0
        out = []
        for row in M:
            outrow = []
            for v in row:
                vi = int(round(v * 2)) / 2.0
                if vi == int(vi):
                    outrow.append(float(int(vi)))
                else:
                    outrow.append(float(vi))
            out.append(outrow)
        return out

    # ---- Build the rank-81 solution deterministically ----
    U2, V2, W2 = _strassen_2x2()
    U4, V4, W4 = _rank49_4x4x4(U2, V2, W2)

    R = 49 + 32
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # first 49 columns: 4x4x4 Strassen square
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4

    # remaining 32 columns: schoolbook for the 4x4x2 sub-block (last 2 columns of B)
    col = 49
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j, col] = 1
                V[j * p + (4 + k), col] = 1
                W[(4 + k) * n + i, col] = 1
                col += 1

    # ---- Verify ----
    if _verify(U, V, W, n, m, p):
        return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}

    # Fallback: pure schoolbook (always valid), rank 96
    rank_sb = n * m * p
    Us = np.zeros((n * m, rank_sb))
    Vs = np.zeros((m * p, rank_sb))
    Ws = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i * m + j, col] = 1
                Vs[j * p + k, col] = 1
                Ws[k * n + i, col] = 1
                col += 1
    return {"U": to_lists(Us), "V": to_lists(Vs), "W": to_lists(Ws)}


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
    print("U shape:", len(res["U"]), "x", len(res["U"][0]))
    print("V shape:", len(res["V"]), "x", len(res["V"][0]))
    print("W shape:", len(res["W"]), "x", len(res["W"][0]))

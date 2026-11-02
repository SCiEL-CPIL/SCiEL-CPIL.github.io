"""Matrix multiplication 2x3x5: block decomposition with Strassen for 2x2x2 cores."""
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2():
    """Return (U,V,W) each shape (4,7) for 2x2x2, index i*2+j / k*2+i."""
    U = [
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
    ]
    W = [
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
    ]
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill
    n, m, p = 2, 3, 5

    # Start from schoolbook (rank 30) as a valid fallback.
    U = [[0.0] * 30 for _ in range(n * m)]
    V = [[0.0] * 30 for _ in range(m * p)]
    W = [[0.0] * 30 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    best = {"U": U, "V": V, "W": W}

    # Build rank-33 decomposition: 15 (row-2 schoolbook) + 7 + 7 + 4 (2x2x5 cores).
    def build():
        U = [[0.0] * 33 for _ in range(n * m)]
        V = [[0.0] * 33 for _ in range(m * p)]
        W = [[0.0] * 33 for _ in range(p * n)]
        c = 0
        # Part A: second row (i=1), 3x5 schoolbook -> 15 terms.
        for j in range(3):
            for k in range(5):
                U[1 * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + 1][c] = 1.0
                c += 1
        # Part B: first row, 2x2x5 = (2x2x2)+(2x2x3).
        sU, sV, sW = _strassen_2x2x2()
        for r in range(7):
            for a in range(4):
                U[a][c + r] = sU[a][r]
                V[a][c + r] = sV[a][r]
                W[a][c + r] = sW[a][r]
        c += 7
        # 2x2x1 (k=2): schoolbook, 4 terms.
        for a in range(4):
            i = a // 2
            j = a % 2
            U[i * m + j][c] = 1.0
            V[j * p + 2][c] = 1.0
            W[2 * n + i][c] = 1.0
            c += 1
        # 2x2x2 (k=3,4): Strassen, 7 terms.
        sU, sV, sW = _strassen_2x2x2()
        for r in range(7):
            for a in range(4):
                i = a // 2
                j = a % 2
                U[i * m + j][c + r] += sU[a][r]
                V[j * p + 3][c + r] += sV[a][r]
                V[j * p + 4][c + r] += sV[a][r]
                W[3 * n + i][c + r] += sW[a][r]
                W[4 * n + i][c + r] += sW[a][r]
        c += 7
        return {"U": U, "V": V, "W": W}, c

    if time.time() < deadline:
        cand, c = build()
        assert c == 33, c
        # Verify the decomposition exactly (integer entries => exact).
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(33):
                        s += cand["U"][i * m + j][r] * cand["V"][j * p + k][r] * cand["W"][k * n + i][r]
                    if s != 1.0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        # Zero entries must remain zero (no accidental off-diagonal products).
        if ok:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(33):
                            s += cand["U"][i * m + j][r] * cand["V"][j * p + k][r] * cand["W"][k * n + i][r]
                        expected = 1.0 if (i == 0 and j == 0 and k == 0) else 0.0
                        # check a sample of off-diagonals fully:
            # Full exact check of all (i,j,k):
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(33):
                            s += cand["U"][i * m + j][r] * cand["V"][j * p + k][r] * cand["W"][k * n + i][r]
                        expected = 1.0 if (i == 0 and j == 0 and k == 0) else 0.0
                        if s != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
        if ok:
            best = cand

    return best

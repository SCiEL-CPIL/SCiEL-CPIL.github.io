"""Fresh design: Strassen (4x4 -> rank 28) + schoolbook (4x4x3 -> rank 12) = 40."""
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start = time.time()
    DEADLINE = 150.0
    SAFETY = 12.0

    def deadline():
        return time.time() + SAFETY >= DEADLINE

    n, m, p = DIMENSIONS
    rank = n * m * p  # 112 schoolbook fallback

    # ---- Schoolbook 4x4x3 (rank 12) ----
    U3 = [[0.0] * 12 for _ in range(16)]
    V3 = [[0.0] * 12 for _ in range(12)]
    W3 = [[0.0] * 12 for _ in range(28)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U3[i * 4 + j][col] = 1.0
                V3[j * 3 + k][col] = 1.0
                W3[k * 4 + i][col] = 1.0
                col += 1

    # ---- Strassen 4x4 (rank 28) ----
    def strassen44():
        # returns (U28, V28, W28) each 16 x 28, indices: U[i*4+j], V[j*4+k], W[k*4+i]
        U = [[0.0] * 28 for _ in range(16)]
        V = [[0.0] * 28 for _ in range(16)]
        W = [[0.0] * 28 for _ in range(16)]
        # 7 bilinear forms at the 2x2 level, each is a sum of 4 (i,j,k) unit tensors
        # (a,b) = (i,j) in A-block, (c,d) = (j,k) in B-block, output (k,i)
        terms = [
            # M1 = (A00+A11)(B00+B11)
            (0, 0, 0, 0, +1), (0, 0, 1, 1, +1), (1, 1, 0, 0, +1), (1, 1, 1, 1, +1),
            # M2 = (A10+A11)B00
            (1, 0, 0, 0, +1), (1, 1, 0, 0, +1),
            # M3 = A00(B10-B00)
            (0, 0, 1, 0, +1), (0, 0, 0, 0, -1),
            # M4 = A11(B01-B11)
            (1, 1, 0, 1, +1), (1, 1, 1, 1, -1),
            # M5 = (A00+A01)B11
            (0, 0, 1, 1, +1), (0, 1, 1, 1, +1),
            # M6 = (A10-A00)(B00+B01)
            (1, 0, 0, 0, +1), (1, 0, 0, 1, +1), (0, 0, 0, 0, -1), (0, 0, 0, 1, -1),
            # M7 = (A01-A11)(B10+B11)
            (0, 1, 1, 0, +1), (0, 1, 1, 1, +1), (1, 1, 1, 0, -1), (1, 1, 1, 1, -1),
        ]
        for (i, j, k, l, s) in terms:
            U[i * 4 + j][col] += s
            V[j * 4 + k][col] += s
            W[k * 4 + l][col] += s
            col += 1
        return U, V, W

    U1, V1, W1 = strassen44()

    # Combine: 40 rank-1 terms. U: 16x40, V: 28x40, W: 28x40
    U = [row[:28] + row[28:] for row in zip(U1, U3)]  # 16 x 40
    V = [row[:28] + row[28:] for row in zip(V1, V3)]  # 28 x 40
    W = [row[:28] + row[28:] for row in zip(W1, W3)]  # 28 x 40

    # ---- Exact verification via sympy ----
    ok = False
    try:
        import sympy
        A = sympy.Matrix(4, 4, [sympy.Symbol(f"a{i}_{j}") for i in range(4) for j in range(4)])
        B = sympy.Matrix(4, 7, [sympy.Symbol(f"b{i}_{j}") for i in range(4) for j in range(7)])
        M = A * B
        # reconstructed
        R = sympy.zeros(4, 7)
        for c in range(40):
            uc = [U[i][c] for i in range(16)]
            vc = [V[j][c] for j in range(28)]
            wc = [W[k][c] for k in range(28)]
            for i in range(4):
                for j in range(4):
                    if uc[i * 4 + j] != 0:
                        for k in range(7):
                            if vc[j * 7 + k] != 0 and wc[k * 4 + i] != 0:
                                R[i, k] += uc[i * 4 + j] * vc[j * 7 + k] * wc[k * 4 + i]
        ok = sympy.simplify(R - M) == sympy.zeros(4, 7)
    except Exception:
        ok = False

    if ok and not deadline():
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 112
    Uf = [[0.0] * rank for _ in range(n * m)]
    Vf = [[0.0] * rank for _ in range(m * p)]
    Wf = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Uf[i * m + j][col] = 1.0
                Vf[j * p + k][col] = 1.0
                Wf[k * n + i][col] = 1.0
                col += 1
    return {"U": Uf, "V": Vf, "W": Wf}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # 2x4 x 4x6 as (2x2)(2x2)(2x2) x (1x2)(2x2)(2x2) x (2x2)(2x2)(2x3)
    # = Strassen(2x2, rank7) x trivial(1x2x2, rank2) x trivial(2x2x3, rank6) => rank 42
    n, m, p = DIMENSIONS

    def d(a, b):
        return 1.0 if a == b else 0.0

    U2, V2, W2 = [], [], []
    for s in range(7):
        u = [0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0]
        w = [0.0, 0.0, 0.0, 0.0]
        for i in range(2):
            for j in range(2):
                u[i * 2 + j] = d(i, 0) + d(j, 0) - d(i, 1) * d(j, 1)
        for j in range(2):
            for k in range(2):
                v[j * 2 + k] = d(j, 1) + d(k, 0) - d(j, 0) * d(k, 1)
        for k in range(2):
            for i in range(2):
                w[k * 2 + i] = d(k, 1) * d(i, 1) + d(k, 0) * d(i, 0)
        U2.append(u); V2.append(v); W2.append(w)

    U1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    V1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    W1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    U3, V3, W3 = [], [], []
    for a in range(2):
        for b in range(2):
            u = [0.0] * 4; u[a * 2 + b] = 1.0; U3.append(u)
            v = [0.0] * 6; v[b * 3 + a] = 1.0; V3.append(v)
            w = [0.0] * 6; w[a * 2 + b] = 1.0; W3.append(w)

    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for s in range(7):
        for q in range(2):
            for r in range(6):
                c = s * 12 + q * 6 + r
                for a in range(2):
                    for b in range(2):
                        U[(a * 2 + b) * 6 + (q * 2 + r)] += U2[s][a * 2 + b] * U1[q * 2 + r] * U3[r][a * 2 + b]
                for b in range(2):
                    for c2 in range(2):
                        V[(b * 2 + c2) * 6 + (q * 2 + r)] += V2[s][b * 2 + c2] * V1[q * 2 + r] * V3[r][b * 3 + c2]
                for c2 in range(2):
                    for a in range(2):
                        W[(c2 * 2 + a) * 2 + (q * 2 + r)] += W2[s][c2 * 2 + a] * W1[q * 2 + r] * W3[r][a * 2 + b]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

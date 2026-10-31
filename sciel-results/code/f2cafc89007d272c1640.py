import time

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _DEADLINE = 145.0

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    beta = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]

    gamma = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]

    delta = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]

    beta_nz = [[(i, j, beta[s][i][j]) for i in range(2) for j in range(2) if beta[s][i][j] != 0] for s in range(7)]
    gamma_nz = [[(j, k, gamma[s][j][k]) for j in range(2) for k in range(2) if gamma[s][j][k] != 0] for s in range(7)]
    delta_nz = [[(i, k, delta[s][i][k]) for i in range(2) for k in range(2) if delta[s][i][k] != 0] for s in range(7)]

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for I, J, cs in beta_nz[s]:
                for ip, jp, ct in beta_nz[t]:
                    i = I * 2 + ip
                    j = J * 2 + jp
                    U[i * 4 + j][col] = cs * ct
            for J, K, cs in gamma_nz[s]:
                for jp, kp, ct in gamma_nz[t]:
                    j = J * 2 + jp
                    k = K * 2 + kp
                    V[j * 4 + k][col] = cs * ct
            for I, K, cs in delta_nz[s]:
                for ip, kp, ct in delta_nz[t]:
                    i = I * 2 + ip
                    k = K * 2 + kp
                    W[k * 4 + i][col] = cs * ct

    if time.time() - _t0 > _DEADLINE:
        raise TimeoutError("Exceeded deadline")
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

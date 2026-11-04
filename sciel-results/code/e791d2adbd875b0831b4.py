"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    idx = 0

    # Strassen on 2x2 blocks for j in {0,1}, k in {0,1}
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1

        a00 = u_idx(0, 0)
        a01 = u_idx(0, 1)
        a10 = u_idx(1, 0)
        a11 = u_idx(1, 1)

        b0k0 = v_idx(0, k0)
        b0k1 = v_idx(0, k1)
        b1k0 = v_idx(1, k0)
        b1k1 = v_idx(1, k1)

        c0k0 = w_idx(k0, 0)
        c0k1 = w_idx(k1, 0)
        c1k0 = w_idx(k0, 1)
        c1k1 = w_idx(k1, 1)

        # M1 = (a00 + a11)(b0k0 + b1k1) -> c0k0, c1k1
        U[a00][idx] += 1
        U[a11][idx] += 1
        V[b0k0][idx] += 1
        V[b1k1][idx] += 1
        W[c0k0][idx] += 1
        W[c1k1][idx] += 1
        idx += 1

        # M2 = (a10 + a11)b0k0 -> c1k0, -c1k1
        U[a10][idx] += 1
        U[a11][idx] += 1
        V[b0k0][idx] += 1
        W[c1k0][idx] += 1
        W[c1k1][idx] -= 1
        idx += 1

        # M3 = a00(b0k1 - b1k1) -> c0k1, c1k1
        U[a00][idx] += 1
        V[b0k1][idx] += 1
        V[b1k1][idx] -= 1
        W[c0k1][idx] += 1
        W[c1k1][idx] += 1
        idx += 1

        # M4 = a11(b1k0 - b0k0) -> c0k0, c1k0
        U[a11][idx] += 1
        V[b1k0][idx] += 1
        V[b0k0][idx] -= 1
        W[c0k0][idx] += 1
        W[c1k0][idx] += 1
        idx += 1

        # M5 = (a00 + a01)b1k1 -> -c0k0, c0k1
        U[a00][idx] += 1
        U[a01][idx] += 1
        V[b1k1][idx] += 1
        W[c0k0][idx] -= 1
        W[c0k1][idx] += 1
        idx += 1

        # M6 = (a10 - a00)(b0k0 + b0k1) -> c1k1
        U[a10][idx] += 1
        U[a00][idx] -= 1
        V[b0k0][idx] += 1
        V[b0k1][idx] += 1
        W[c1k1][idx] += 1
        idx += 1

        # M7 = (a01 - a11)(b1k0 + b1k1) -> c0k0
        U[a01][idx] += 1
        U[a11][idx] -= 1
        V[b1k0][idx] += 1
        V[b1k1][idx] += 1
        W[c0k0][idx] += 1
        idx += 1

    # Schoolbook for j = 2
    for i in range(n):
        for k in range(p):
            U[u_idx(i, 2)][idx] += 1
            V[v_idx(2, k)][idx] += 1
            W[w_idx(k, i)][idx] += 1
            idx += 1

    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END

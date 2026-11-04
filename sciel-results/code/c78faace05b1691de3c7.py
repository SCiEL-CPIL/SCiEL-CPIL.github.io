import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Strassen-based decomposition for rank 55
    def make_strassen():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1

            U[0 * 3 + 0][col] += 1
            U[1 * 3 + 1][col] += 1
            V[0 * 10 + k0][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            U[1 * 3 + 0][col] += 1
            U[1 * 3 + 1][col] += 1
            V[0 * 10 + k0][col] += 1
            W[k0 * 2 + 1][col] += 1
            W[k1 * 2 + 1][col] -= 1
            col += 1

            U[0 * 3 + 0][col] += 1
            V[0 * 10 + k1][col] += 1
            V[1 * 10 + k1][col] -= 1
            W[k1 * 2 + 0][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            U[1 * 3 + 1][col] += 1
            V[1 * 10 + k0][col] += 1
            V[0 * 10 + k0][col] -= 1
            W[k0 * 2 + 0][col] += 1
            W[k0 * 2 + 1][col] += 1
            col += 1

            U[0 * 3 + 0][col] += 1
            U[0 * 3 + 1][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] -= 1
            W[k1 * 2 + 0][col] += 1
            col += 1

            U[1 * 3 + 0][col] += 1
            U[0 * 3 + 0][col] -= 1
            V[0 * 10 + k0][col] += 1
            V[0 * 10 + k1][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

            U[0 * 3 + 1][col] += 1
            U[1 * 3 + 1][col] -= 1
            V[1 * 10 + k0][col] += 1
            V[1 * 10 + k1][col] += 1
            W[k0 * 2 + 0][col] += 1
            col += 1

            U[0 * 3 + 2][col] += 1
            V[2 * 10 + k0][col] += 1
            W[k0 * 2 + 0][col] += 1
            col += 1

            U[0 * 3 + 2][col] += 1
            V[2 * 10 + k1][col] += 1
            W[k1 * 2 + 0][col] += 1
            col += 1

            U[1 * 3 + 2][col] += 1
            V[2 * 10 + k0][col] += 1
            W[k0 * 2 + 1][col] += 1
            col += 1

            U[1 * 3 + 2][col] += 1
            V[2 * 10 + k1][col] += 1
            W[k1 * 2 + 1][col] += 1
            col += 1

        assert col == rank, f"Expected rank {rank}, got {col}"
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        return True

    try:
        result = make_strassen()
        if validate(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass

    return make_schoolbook()

import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 145.0  # 150s wall-clock limit, 5s safety margin

    n, m, p = 2, 3, 6
    rank = 33

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    # Strassen factors for 2x2x2
    S_U = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    S_V = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    S_W = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]

    # Standard factors for 2x1x2
    Std_U = [
        np.array([[1], [0]]),
        np.array([[1], [0]]),
        np.array([[0], [1]]),
        np.array([[0], [1]])
    ]
    Std_V = [
        np.array([[1, 0]]),
        np.array([[0, 1]]),
        np.array([[1, 0]]),
        np.array([[0, 1]])
    ]
    Std_W = [
        np.array([[1, 0], [0, 0]]),   # C00
        np.array([[0, 0], [1, 0]]),   # C01
        np.array([[0, 1], [0, 0]]),   # C10
        np.array([[0, 0], [0, 1]])    # C11
    ]

    col = 0
    for b in range(3):
        # Strassen part: A[:, 0:2] * B[0:2, 2b:2b+2]
        for r in range(7):
            if time.time() > _deadline:
                # Return best complete result so far; since work is O(1)
                # this branch is effectively unreachable but guards against
                # pathological scheduling delays.
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i, j]
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j, kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp, i]
            col += 1

        # Standard part: A[:, 2] * B[2, 2b:2b+2]
        for r in range(4):
            if time.time() > _deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            for i in range(2):
                U[i * m + 2, col] = Std_U[r][i, 0]
            for kp in range(2):
                k_global = 2 * b + kp
                V[2 * p + k_global, col] = Std_V[r][0, kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = Std_W[r][kp, i]
            col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

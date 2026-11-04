import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the rank-42 Strassen decomposition
    def build_strassen_42():
        rank = 42
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        def add_strassen_block(i_start, j_start, k_start, offset):
            a00 = (i_start + 0) * m + (j_start + 0)
            a01 = (i_start + 0) * m + (j_start + 1)
            a10 = (i_start + 1) * m + (j_start + 0)
            a11 = (i_start + 1) * m + (j_start + 1)

            b00 = (j_start + 0) * p + (k_start + 0)
            b01 = (j_start + 0) * p + (k_start + 1)
            b10 = (j_start + 1) * p + (k_start + 0)
            b11 = (j_start + 1) * p + (k_start + 1)

            w00 = (k_start + 0) * n + (i_start + 0)
            w01 = (k_start + 1) * n + (i_start + 0)
            w10 = (k_start + 0) * n + (i_start + 1)
            w11 = (k_start + 1) * n + (i_start + 1)

            U[a00, offset] += 1.0
            U[a11, offset] += 1.0
            V[b00, offset] += 1.0
            V[b11, offset] += 1.0
            W[w00, offset] += 1.0
            W[w11, offset] += 1.0

            U[a10, offset + 1] += 1.0
            U[a11, offset + 1] += 1.0
            V[b00, offset + 1] += 1.0
            W[w10, offset + 1] += 1.0
            W[w11, offset + 1] -= 1.0

            U[a00, offset + 2] += 1.0
            V[b01, offset + 2] += 1.0
            V[b11, offset + 2] -= 1.0
            W[w01, offset + 2] += 1.0
            W[w11, offset + 2] += 1.0

            U[a11, offset + 3] += 1.0
            V[b10, offset + 3] += 1.0
            V[b00, offset + 3] -= 1.0
            W[w00, offset + 3] += 1.0
            W[w10, offset + 3] += 1.0

            U[a00, offset + 4] += 1.0
            U[a01, offset + 4] += 1.0
            V[b11, offset + 4] += 1.0
            W[w00, offset + 4] -= 1.0
            W[w01, offset + 4] += 1.0

            U[a10, offset + 5] += 1.0
            U[a00, offset + 5] -= 1.0
            V[b00, offset + 5] += 1.0
            V[b01, offset + 5] += 1.0
            W[w11, offset + 5] += 1.0

            U[a01, offset + 6] += 1.0
            U[a11, offset + 6] -= 1.0
            V[b10, offset + 6] += 1.0
            V[b11, offset + 6] += 1.0
            W[w00, offset + 6] += 1.0

        offset = 0
        for i_start in range(0, n, 2):
            for j_start in range(0, m, 2):
                for k_start in range(0, p, 2):
                    add_strassen_block(i_start, j_start, k_start, offset)
                    offset += 7

        return U, V, W, rank

    U42, V42, W42, r42 = build_strassen_42()

    # Return the rank-42 decomposition
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}

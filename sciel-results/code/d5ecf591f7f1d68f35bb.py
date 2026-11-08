"""Matrix multiplication 2x5x6 tensor decomposition - rank 54."""
import numpy as np
import time

N_DIM, M_DIM, P_DIM = 2, 5, 6
TARGET_RANK = 54


def _build_factors():
    n, m, p = N_DIM, M_DIM, P_DIM
    r = TARGET_RANK

    U_mat = [[0.0] * r for _ in range(n * m)]
    V_mat = [[0.0] * r for _ in range(m * p)]
    W_mat = [[0.0] * r for _ in range(p * n)]

    idx = 0

    jpairs = ((0, 1), (2, 3))
    kpairs = ((0, 1), (2, 3), (4, 5))

    for (ja, jb) in jpairs:
        for (ka, kb) in kpairs:
            # Term 1: (A[0,ja] + A[1,jb]) * (B[ja,ka] + B[jb,kb])
            U_mat[ja][idx] = 1.0
            U_mat[m + jb][idx] = 1.0
            V_mat[ja * p + ka][idx] = 1.0
            V_mat[jb * p + kb][idx] = 1.0
            W_mat[ka * n][idx] = 1.0
            W_mat[kb * n + 1][idx] = 1.0
            idx += 1

            # Term 2: (A[1,ja] + A[1,jb]) * B[ja,ka]
            U_mat[m + ja][idx] = 1.0
            U_mat[m + jb][idx] = 1.0
            V_mat[ja * p + ka][idx] = 1.0
            W_mat[ka * n + 1][idx] = 1.0
            W_mat[kb * n + 1][idx] = -1.0
            idx += 1

            # Term 3: A[0,ja] * (B[ja,kb] - B[jb,kb])
            U_mat[ja][idx] = 1.0
            V_mat[ja * p + kb][idx] = 1.0
            V_mat[jb * p + kb][idx] = -1.0
            W_mat[kb * n][idx] = 1.0
            W_mat[kb * n + 1][idx] = 1.0
            idx += 1

            # Term 4: A[1,jb] * (B[jb,ka] - B[ja,ka])
            U_mat[m + jb][idx] = 1.0
            V_mat[jb * p + ka][idx] = 1.0
            V_mat[ja * p + ka][idx] = -1.0
            W_mat[ka * n][idx] = 1.0
            W_mat[ka * n + 1][idx] = 1.0
            idx += 1

            # Term 5: (A[0,ja] + A[0,jb]) * B[jb,kb]
            U_mat[ja][idx] = 1.0
            U_mat[jb][idx] = 1.0
            V_mat[jb * p + kb][idx] = 1.0
            W_mat[ka * n][idx] = -1.0
            W_mat[kb * n][idx] = 1.0
            idx += 1

            # Term 6: (A[1,ja] - A[0,ja]) * (B[ja,ka] + B[ja,kb])
            U_mat[m + ja][idx] = 1.0
            U_mat[ja][idx] = -1.0
            V_mat[ja * p + ka][idx] = 1.0
            V_mat[ja * p + kb][idx] = 1.0
            W_mat[kb * n + 1][idx] = 1.0
            idx += 1

            # Term 7: (A[0,jb] - A[1,jb]) * (B[jb,ka] + B[jb,kb])
            U_mat[jb][idx] = 1.0
            U_mat[m + jb][idx] = -1.0
            V_mat[jb * p + ka][idx] = 1.0
            V_mat[jb * p + kb][idx] = 1.0
            W_mat[ka * n][idx] = 1.0
            idx += 1

    # Handle j=4 directly
    for kk in range(P_DIM):
        U_mat[4][idx] = 1.0
        V_mat[4 * p + kk][idx] = 1.0
        W_mat[kk * n][idx] = 1.0
        idx += 1

        U_mat[m + 4][idx] = 1.0
        V_mat[4 * p + kk][idx] = 1.0
        W_mat[kk * n + 1][idx] = 1.0
        idx += 1

    assert idx == r, f"Expected {r} terms, got {idx}"
    return U_mat, V_mat, W_mat


def construct():
    t0 = time.perf_counter()
    limit = t0 + 115.0

    if time.perf_counter() > limit:
        raise RuntimeError("timeout before start")

    Uf, Vf, Wf = _build_factors()

    if time.perf_counter() > limit:
        raise RuntimeError("timeout after build")

    return {"U": Uf, "V": Vf, "W": Wf}

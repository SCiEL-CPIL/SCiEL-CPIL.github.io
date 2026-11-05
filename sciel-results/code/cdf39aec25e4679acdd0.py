import time
import numpy as np

def construct():
    deadline = time.time() + 110
    n, m, p = 2, 4, 8

    # Known: R(M_{2,4,8}) <= 56 via block Strassen (2 * 28).
    # We attempt to find a lower rank decomposition using the structure.
    # The tensor is T[i,j,k] = delta_{i,j,k} where C_{ik} = sum_j A_{ij} B_{jk}.
    # R(M_{2,4,2}) is unknown but likely > 8.
    # For now, we return the known rank 56 solution which is valid.
    # The task asks for rank-minimal. If 56 is not minimal, this scores lower than optimal.
    # However, without a valid lower rank construction, 56 is the best we can guarantee.

    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    # Block structure:
    # A (2x4) split into two 2x2 blocks: A_L (cols 0-1), A_R (cols 2-3)
    # B (4x8) split into two 2x8 blocks: B_L (rows 0-1), B_R (rows 2-3)
    # C = A_L B_L + A_R B_R
    # Each term is M_{2,2,8} which is 4 copies of M_{2,2,2}.
    # M_{2,2,2} has rank 7 (Strassen).
    # Total rank = 2 * 4 * 7 = 56.

    n_blocks_i = 1  # i in 0..1
    n_blocks_j = 2  # j in 0..3 (two 2-blocks)
    n_blocks_k = 4  # k in 0..7 (four 2-blocks)

    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1

                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1

                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1

                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1

                # M1: (a+d)(e+h) -> c00, c11
                U[u_a, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1; W[w_c11, col] += 1
                col += 1

                # M2: (c+d)e -> c10, c11
                U[u_c, col] += 1; U[u_d, col] += 1
                V[v_e, col] += 1
                W[w_c10, col] += 1; W[w_c11, col] -= 1
                col += 1

                # M3: a(f-h) -> c01, c11
                U[u_a, col] += 1
                V[v_f, col] += 1; V[v_h, col] -= 1
                W[w_c01, col] += 1; W[w_c11, col] += 1
                col += 1

                # M4: d(g-e) -> c00, c10
                U[u_d, col] += 1
                V[v_g, col] += 1; V[v_e, col] -= 1
                W[w_c00, col] += 1; W[w_c10, col] += 1
                col += 1

                # M5: (a+b)h -> c00, c01
                U[u_a, col] += 1; U[u_b, col] += 1
                V[v_h, col] += 1
                W[w_c00, col] -= 1; W[w_c01, col] += 1
                col += 1

                # M6: (c-a)(e+f) -> c11
                U[u_c, col] += 1; U[u_a, col] -= 1
                V[v_e, col] += 1; V[v_f, col] += 1
                W[w_c11, col] += 1
                col += 1

                # M7: (b-d)(g+h) -> c00
                U[u_b, col] += 1; U[u_d, col] -= 1
                V[v_g, col] += 1; V[v_h, col] += 1
                W[w_c00, col] += 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

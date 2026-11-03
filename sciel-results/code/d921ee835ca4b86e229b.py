"""Matrix multiplication tensor decomposition for 3x4x8."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses Strassen's algorithm for 2x2x2 blocks where possible, and
    schoolbook for the remaining 1x4x8 part.
    """
    n, m, p = 3, 4, 8
    U = np.zeros((n * m, 0))  # 12 x rank
    V = np.zeros((m * p, 0))  # 32 x rank
    W = np.zeros((p * n, 0))  # 24 x rank

    def add_strassen_block(i_base, j_base, k_base):
        """
        Add a Strassen 2x2x2 block.
        Returns 7 new rank terms as columns of (U, V, W).
        """
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        u1 = np.zeros(n * m)
        v1 = np.zeros(m * p)
        w1 = np.zeros(p * n)
        u1[i_base * m + j_base] = 1
        u1[(i_base + 1) * m + (j_base + 1)] = 1
        v1[j_base * p + k_base] = 1
        v1[(j_base + 1) * p + (k_base + 1)] = 1
        w1[k_base * n + i_base] = 1
        w1[(k_base + 1) * n + (i_base + 1)] = 1

        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        u2 = np.zeros(n * m)
        v2 = np.zeros(m * p)
        w2 = np.zeros(p * n)
        u2[(i_base + 1) * m + j_base] = 1
        u2[(i_base + 1) * m + (j_base + 1)] = 1
        v2[j_base * p + k_base] = 1
        w2[k_base * n + (i_base + 1)] = 1
        w2[(k_base + 1) * n + (i_base + 1)] = 1

        # M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
        u3 = np.zeros(n * m)
        v3 = np.zeros(m * p)
        w3 = np.zeros(p * n)
        u3[i_base * m + j_base] = 1
        v3[(j_base + 1) * p + k_base] = 1
        v3[(j_base + 1) * p + (k_base + 1)] = -1
        w3[k_base * n + i_base] = 1
        w3[(k_base + 1) * n + i_base] = -1

        # M4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
        u4 = np.zeros(n * m)
        v4 = np.zeros(m * p)
        w4 = np.zeros(p * n)
        u4[(i_base + 1) * m + (j_base + 1)] = 1
        v4[j_base * p + (k_base + 1)] = 1
        v4[j_base * p + k_base] = -1
        w4[(k_base + 1) * n + (i_base + 1)] = 1
        w4[k_base * n + (i_base + 1)] = -1

        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        u5 = np.zeros(n * m)
        v5 = np.zeros(m * p)
        w5 = np.zeros(p * n)
        u5[i_base * m + j_base] = 1
        u5[i_base * m + (j_base + 1)] = 1
        v5[(j_base + 1) * p + (k_base + 1)] = 1
        w5[(k_base + 1) * n + i_base] = 1
        w5[(k_base + 1) * n + (i_base + 1)] = 1

        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        u6 = np.zeros(n * m)
        v6 = np.zeros(m * p)
        w6 = np.zeros(p * n)
        u6[(i_base + 1) * m + j_base] = 1
        u6[i_base * m + j_base] = -1
        v6[j_base * p + k_base] = 1
        v6[j_base * p + (k_base + 1)] = 1
        w6[k_base * n + (i_base + 1)] = 1
        w6[(k_base + 1) * n + (i_base + 1)] = 1

        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        u7 = np.zeros(n * m)
        v7 = np.zeros(m * p)
        w7 = np.zeros(p * n)
        u7[i_base * m + (j_base + 1)] = 1
        u7[(i_base + 1) * m + (j_base + 1)] = -1
        v7[(j_base + 1) * p + k_base] = 1
        v7[(j_base + 1) * p + (k_base + 1)] = 1
        w7[k_base * n + i_base] = 1
        w7[k_base * n + (i_base + 1)] = 1

        U_block = np.column_stack([u1, u2, u3, u4, u5, u6, u7])
        V_block = np.column_stack([v1, v2, v3, v4, v5, v6, v7])
        W_block = np.column_stack([w1, w2, w3, w4, w5, w6, w7])
        return U_block, V_block, W_block

    # Group 1: rows 0, 1 (2x4x8)
    # Split 4 columns into two groups of 2: j_base = 0, 2
    # Split 8 columns into four groups of 2: k_base = 0, 2, 4, 6
    all_U = []
    all_V = []
    all_W = []

    for j_base in [0, 2]:
        for k_base in [0, 2, 4, 6]:
            Ub, Vb, Wb = add_strassen_block(0, j_base, k_base)
            all_U.append(Ub)
            all_V.append(Vb)
            all_W.append(Wb)

    # Group 2: row 2 (1x4x8) - schoolbook
    for j in range(4):
        for k in range(8):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            u[2 * m + j] = 1
            v[j * p + k] = 1
            w[k * n + 2] = 1
            all_U.append(u.reshape(-1, 1))
            all_V.append(v.reshape(-1, 1))
            all_W.append(w.reshape(-1, 1))

    U_final = np.hstack(all_U)
    V_final = np.hstack(all_V)
    W_final = np.hstack(all_W)

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

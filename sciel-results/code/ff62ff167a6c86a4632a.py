"""Matrix multiplication 2x4x6 rank-minimal decomposition via Strassen block structure."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    t0 = time.time()
    deadline = t0 + 110

    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U_strassen = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]

    V_strassen = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]

    W_strassen = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    col = 0
    for b in range(3):
        for a in range(2):
            for s in range(7):
                for (idx, coeff) in U_strassen[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    U[i_loc * m + a * 2 + j_loc, col] += coeff
                for (idx, coeff) in V_strassen[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    V[(a * 2 + j_loc) * p + b * 2 + k_loc, col] += coeff
                for (idx, coeff) in W_strassen[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    W[(b * 2 + k_loc) * n + i_loc, col] += coeff
                col += 1

    def build_target():
        T = np.zeros((n * m, m * p, p * n))
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    T[i_A * m + j, j * p + k_B, k_B * n + i_A] = 1.0
        return T

    T_target = build_target()
    T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_comp, T_target, atol=1e-9):
        # Fallback: schoolbook rank-48
        rank0 = n * m * p
        U = np.zeros((n * m, rank0))
        V = np.zeros((m * p, rank0))
        W = np.zeros((p * n, rank0))
        r = 0
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    U[i_A * m + j, r] = 1.0
                    V[j * p + k_B, r] = 1.0
                    W[k_B * n + i_A, r] = 1.0
                    r += 1

    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    T_comp2 = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_comp2, T_target, atol=1e-9):
        # Rebuild unrounded
        U = np.zeros((n * m, 42))
        V = np.zeros((m * p, 42))
        W = np.zeros((p * n, 42))
        col = 0
        for b in range(3):
            for a in range(2):
                for s in range(7):
                    for (idx, coeff) in U_strassen[s]:
                        U[idx // 2 * m + a * 2 + idx % 2, col] += coeff
                    for (idx, coeff) in V_strassen[s]:
                        V[(a * 2 + idx // 2) * p + b * 2 + idx % 2, col] += coeff
                    for (idx, coeff) in W_strassen[s]:
                        W[(b * 2 + idx % 2) * n + idx // 2, col] += coeff
                    col += 1

    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }

# EVOLVE-BLOCK-END

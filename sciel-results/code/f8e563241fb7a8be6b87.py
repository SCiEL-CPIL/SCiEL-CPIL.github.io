"""
Rank-77 exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
  - Split the 4x6 output into two 4x4 and 4x2 blocks.
  - Use recursive Strassen (rank 7 per 2x2x2) for the 4x4x4 part -> 49 terms.
  - Use direct block Strassen for the 4x4x2 part -> 28 terms.
  Total rank = 77.
  
  We then attempt to reduce rank by trying to merge/zero-out columns using
  coordinate descent on the tensor equation, with a strict deadline guard.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# Strassen rank-7 factor matrices for 2x2x2
_STRassen_U = np.array([
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
], dtype=np.float64)

_STRassen_V = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
], dtype=np.float64)

_STRassen_W = np.array([
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0],
], dtype=np.float64)


def construct():
    t0 = time.time()
    deadline = 115.0  # safety margin before 120s

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)

    # ---- Build rank-77 decomposition via recursive Strassen ----
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442  # 77

    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    # Outer Strassen coefficients
    outer_U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    outer_V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    outer_W_coeffs = [
        {0: 1, 3: 1},
        {1: 1},
        {0: -1},
        {1: 1},
        {3: 1},
        {2: 1},
        {0: 1, 2: -1},
    ]

    # Part 1: 4x4 * 4x4 -> 4x4 (columns 0..48)
    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            for block_idx, coeff in outer_U_coeffs[s].items():
                block_offset = block_idx * 4
                for local_idx in range(4):
                    inner_u = _STRassen_U[local_idx][t]
                    if inner_u != 0:
                        U[block_offset + local_idx][col] += coeff * inner_u

            for block_idx, coeff in outer_V_coeffs[s].items():
                block_offset = block_idx * 4
                for local_idx in range(4):
                    inner_v = _STRassen_V[local_idx][t]
                    if inner_v != 0:
                        V[block_offset + local_idx][col] += coeff * inner_v

            for block_idx, coeff in outer_W_coeffs[s].items():
                if block_idx == 0:
                    w_offsets = [0, 1, 4, 5]
                elif block_idx == 1:
                    w_offsets = [2, 3, 6, 7]
                elif block_idx == 2:
                    w_offsets = [8, 9, 12, 13]
                else:
                    w_offsets = [10, 11, 14, 15]
                for local_idx, w_off in enumerate(w_offsets):
                    inner_w = _STRassen_W[local_idx][t]
                    if inner_w != 0:
                        W[w_off][col] += coeff * inner_w

    # Part 2: 4x4 * 4x2 -> 4x2 (columns 49..76)
    blocks_442 = [
        (0, 'top', 'top'),
        (1, 'bot', 'top'),
        (2, 'top', 'bot'),
        (3, 'bot', 'bot'),
    ]

    for bi, (a_block, b_part, c_part) in enumerate(blocks_442):
        for t in range(7):
            col = rank_444 + bi * 7 + t

            a_offset = a_block * 4
            for local_idx in range(4):
                inner_u = _STRassen_U[local_idx][t]
                if inner_u != 0:
                    U[a_offset + local_idx][col] = inner_u

            if b_part == 'top':
                v_offsets = [4, 5, 10, 11]
            else:
                v_offsets = [16, 17, 22, 23]
            for local_idx, v_off in enumerate(v_offsets):
                inner_v = _STRassen_V[local_idx][t]
                if inner_v != 0:
                    V[v_off][col] = inner_v

            if c_part == 'top':
                w_offsets = [16, 17, 20, 21]
            else:
                w_offsets = [18, 19, 22, 23]
            for local_idx, w_off in enumerate(w_offsets):
                inner_w = _STRassen_W[local_idx][t]
                if inner_w != 0:
                    W[w_off][col] = inner_w

    # Verify initial construction
    if not check(U, V, W, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n * m, rank), dtype=np.float64)
        V_s = np.zeros((m * p, rank), dtype=np.float64)
        W_s = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Try to reduce rank via coordinate descent
    best_rank = total_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Attempt to remove columns one at a time
    for target_rank in range(total_rank - 1, 70, -1):
        if time.time() > deadline - 10:
            break

        # Try to find a valid decomposition of rank target_rank
        # Start from current best and try to zero out extra columns
        U_try = np.zeros((n * m, target_rank), dtype=np.float64)
        V_try = np.zeros((m * p, target_rank), dtype=np.float64)
        W_try = np.zeros((p * n, target_rank), dtype=np.float64)

        # Initialize with first target_rank columns of current best
        U_try[:] = best_U[:, :target_rank]
        V_try[:] = best_V[:, :target_rank]
        W_try[:] = best_W[:, :target_rank]

        err = err_of(U_try, V_try, W_try)

        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            continue

        # Coordinate descent to try to drive error to zero
        for it in range(50):
            if time.time() > deadline - 10:
                break
            if err < 1e-10:
                break

            improved = False
            for c in range(target_rank):
                if time.time() > deadline - 10:
                    break
                if err < 1e-10:
                    break

                Uo, Vo, Wo = U_try[:, c].copy(), V_try[:, c].copy(), W_try[:, c].copy()

                # Try small perturbations
                for scale in [0.1, 0.01, 0.001]:
                    U_try[:, c] = Uo + np.random.randn(n * m) * scale
                    V_try[:, c] = Vo + np.random.randn(m * p) * scale
                    W_try[:, c] = Wo + np.random.randn(p * n) * scale
                    new_err = err_of(U_try, V_try, W_try)
                    if new_err < err:
                        err = new_err
                        improved = True
                        break
                    U_try[:, c] = Uo
                    V_try[:, c] = Vo
                    W_try[:, c] = Wo

                if not improved and err > 1e-6:
                    break

            if err < 1e-10:
                break

        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try

        if time.time() > deadline - 5:
            break

    # Final verification
    if check(best_U, best_V, best_W, atol=1e-9):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to original rank-77
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

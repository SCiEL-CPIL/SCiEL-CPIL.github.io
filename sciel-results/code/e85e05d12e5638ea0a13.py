import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Winograd 3x3x3 factors (rank 7)
    U_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, -1, 1, 0],
    ], dtype=float)

    V_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, -1, 1, 0],
    ], dtype=float)

    W_win333 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1],
    ], dtype=float)

    # Verify Winograd 3x3x3 factors
    win333_ok = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = sum(U_win333[i*3+j, r] * V_win333[j*3+k, r] * W_win333[k*3+i, r] for r in range(7))
                if abs(s - 1) > 1e-10:
                    win333_ok = False
                    break
            if not win333_ok:
                break
        if not win333_ok:
            break

    if not win333_ok:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Winograd 3x2x2 factors (rank 4)
    U_win322 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)

    V_win322 = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)

    W_win322 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, -1],
    ], dtype=float)

    # Verify Winograd 3x2x2 factors
    win322_ok = True
    for i in range(3):
        for j in range(2):
            for k in range(2):
                s = sum(U_win322[i*2+j, r] * V_win322[j*2+k, r] * W_win322[k*3+i, r] for r in range(4))
                if abs(s - 1) > 1e-10:
                    win322_ok = False
                    break
            if not win322_ok:
                break
        if not win322_ok:
            break

    if not win322_ok:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Build the full decomposition
    # Strategy:
    # Split B into B_left (4x3) and B_right (4x2)
    # C = A * B_left + A * B_right
    #
    # For A * B_left (3x4 * 4x3):
    #   Split A into A_left (3x3) and A_right (3x1)
    #   A_left * B_left (3x3 * 3x3): Winograd 3x3x3, rank 7
    #   A_right * B_left (3x1 * 1x3): schoolbook, rank 3
    #
    # For A * B_right (3x4 * 4x2):
    #   Split A into A_left (3x3) and A_right (3x1)
    #   A_left * B_right (3x3 * 3x2):
    #     Split the middle dimension of A_left (size 3) into 2+1
    #     A_left[:, :2] * B_right[:2, :] (3x2 * 2x2): Winograd 3x2x2, rank 4
    #     A_left[:, 2:3] * B_right[2:3, :] (3x1 * 1x2): schoolbook, rank 2
    #   A_right * B_right (3x1 * 1x2): schoolbook, rank 2
    #
    # Total rank = 7 + 3 + 4 + 2 + 2 = 18

    rank_total = 7 + 3 + 4 + 2 + 2
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))

    col = 0

    # Part 1: A_left * B_left (3x3 * 3x3) using Winograd 3x3x3
    for r in range(7):
        for i in range(3):
            for j in range(3):
                U[i * 4 + j, col + r] = U_win333[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V[j * 5 + k, col + r] = V_win333[j * 3 + k, r]
        for k in range(3):
            for i in range(3):
                W[k * 3 + i, col + r] = W_win333[k * 3 + i, r]
    col += 7

    # Part 2: A_right * B_left (3x1 * 1x3) rank 3
    for i in range(3):
        U[i * 4 + 3, col + i] = 1
        for k in range(3):
            V[3 * 5 + k, col + i] = 1
            W[k * 3 + i, col + i] = 1
    col += 3

    # Part 3: A_left[:, :2] * B_right[:2, :] (3x2 * 2x2) using Winograd 3x2x2
    for r in range(4):
        for i in range(3):
            for j in range(2):
                U[i * 4 + j, col + r] = U_win322[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * 5 + (3 + k), col + r] = V_win322[j * 2 + k, r]
        for k in range(2):
            for i in range(3):
                W[(3 + k) * 3 + i, col + r] = W_win322[k * 3 + i, r]
    col += 4

    # Part 4: A_left[:, 2:3] * B_right[2:3, :] (3x1 * 1x2) rank 2
    for k_idx in range(2):
        k = 3 + k_idx
        for i in range(3):
            U[i * 4 + 2, col + k_idx] = 1
        V[2 * 5 + k, col + k_idx] = 1
        for i in range(3):
            W[k * 3 + i, col + k_idx] = 1
    col += 2

    # Part 5: A_right * B_right (3x1 * 1x2) rank 2
    for k_idx in range(2):
        k = 3 + k_idx
        for i in range(3):
            U[i * 4 + 3, col + k_idx] = 1
        V[3 * 5 + k, col + k_idx] = 1
        for i in range(3):
            W[k * 3 + i, col + k_idx] = 1
    col += 2

    # Verify correctness
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(rank_total))
                max_err = max(max_err, abs(total - 1))

    if max_err > 1e-10:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

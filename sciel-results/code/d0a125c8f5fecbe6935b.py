import time
import sys
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        u_val = u_row[c]
                        v_val = v_row[c]
                        w_val = w_row[c]
                        if u_val == 0 or v_val == 0 or w_val == 0:
                            continue
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build 36-term candidate: two 2x3x4 blocks, each of rank 18
    # Each 2x3x4 block: 2 Strassen 2x2x2 (rank 7 each) + 2 trivial 2x1x2 (rank 2 each)
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for half in range(2):
        k_offset = half * 4  # 0 or 4
        # Sub-block 1: A[:,0:2] * B[0:2, k_offset:k_offset+2] -> C[:, k_offset:k_offset+2]
        # This is 2x2x2, use Strassen
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] += u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + k_offset + lk][col] += v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k_offset + lk) * n + li][col] += w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

        # Sub-block 2: A[:,0:2] * B[0:2, k_offset+2:k_offset+4] -> C[:, k_offset+2:k_offset+4]
        # This is 2x2x2, use Strassen
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] += u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + k_offset + 2 + lk][col] += v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k_offset + 2 + lk) * n + li][col] += w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

        # Sub-block 3: A[:,2:3] * B[2:3, k_offset:k_offset+2] -> C[:, k_offset:k_offset+2]
        # This is 2x1x2, trivial rank 2
        for lk in range(2):
            for li in range(2):
                U[li * m + 2][col] = 1
                V[2 * p + k_offset + lk][col] = 1
                W[(k_offset + lk) * n + li][col] = 1
                col += 1
            if not _elapsed_ok():
                return _schoolbook()

        # Sub-block 4: A[:,2:3] * B[2:3, k_offset+2:k_offset+4] -> C[:, k_offset+2:k_offset+4]
        # This is 2x1x2, trivial rank 2
        for lk in range(2):
            for li in range(2):
                U[li * m + 2][col] = 1
                V[2 * p + k_offset + 2 + lk][col] = 1
                W[(k_offset + 2 + lk) * n + li][col] = 1
                col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # p2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # p3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # p4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # p5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                if val != 0:
                                    U4[i * 4 + j][r] += val
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                if val != 0:
                                    V4[j * 4 + k][r] += val
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                if val != 0:
                                    W4[k * 4 + i][r] += val
        return U4, V4, W4

    def build_4x4x7_decomposition():
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j * 7 + k][r] = V4[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k * 4 + i][r] = W4[k * 4 + i][r]
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i * 4 + j][col_offset] = 1
                    V[j * 7 + k][col_offset] = 1
                    W[k * 4 + i][col_offset] = 1
                    col_offset += 1
        return U, V, W

    def tensors_to_vec(U, V, W):
        r = len(U[0])
        u = np.array(U, dtype=float).flatten()
        v = np.array(V, dtype=float).flatten()
        w = np.array(W, dtype=float).flatten()
        return np.concatenate([u, v, w])

    def vec_to_tensors(x, r):
        u_len = n * m * r
        v_len = m * p * r
        U = x[:u_len].reshape(n * m, r)
        V = x[u_len:u_len + v_len].reshape(m * p, r)
        W = x[u_len + v_len:].reshape(p * n, r)
        return U, V, W

    def build_jacobian(U, V, W):
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        u_len = n * m * r
        v_len = m * p * r
        J = np.zeros((n * m * p, u_len + v_len + p * n * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i * m + j) * p + k
                    for c in range(r):
                        uv = V[j * p + k, c] * W[k * n + i, c]
                        vw = U[i * m + j, c] * W[k * n + i, c]
                        wu = U[i * m + j, c] * V[j * p + k, c]
                        J[row, i * m * r + j * r + c] = uv
                        J[row, u_len + j * p * r + k * r + c] = vw
                        J[row, u_len + v_len + k * n * r + i * r + c] = wu
        return J

    def residual_vec(U, V, W):
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        res = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    res[(i * m + j) * p + k] = s - 1.0
        return res

    def round_to_half_integers(U, V, W):
        U_r = np.round(np.array(U) * 2) / 2
        V_r = np.round(np.array(V) * 2) / 2
        W_r = np.round(np.array(W) * 2) / 2
        return U_r, V_r, W_r

    def random_merge(U, V, W, n_merge, rng, tol=1e-9):
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        new_r = r - n_merge
        cols = list(range(r))
        rng.shuffle(cols)
        kept = cols[:new_r]
        dropped = cols[new_r:]
        M = np.zeros((new_r, r))
        for t, kc in enumerate(kept):
            M[t, kc] = 1.0
            for dc in dropped:
                M[t, dc] = rng.gauss(0, 0.5)

        U_new = U @ M.T
        V_new = V @ M.T
        W_new = W @ M.T

        res = residual_vec(U_new, V_new, W_new)
        J = build_jacobian(U_new, V_new, W_new)
        try:
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except Exception:
            return None
        x_new = tensors_to_vec(U_new, V_new, W_new) + delta
        U2, V2, W2 = vec_to_tensors(x_new, new_r)

        if not check_valid(U2.tolist(), V2.tolist(), W2.tolist(), tol=1e-8):
            # Try a second-order Newton step
            res2 = residual_vec(U2, V2, W2)
            if np.max(np.abs(res2)) < 1e-6:
                J2 = build_jacobian(U2, V2, W2)
                try:
                    delta2, *_ = np.linalg.lstsq(J2, -res2, rcond=None)
                    x2 = tensors_to_vec(U2, V2, W2) + delta2
                    U3, V3, W3 = vec_to_tensors(x2, new_r)
                    if check_valid(U3.tolist(), V3.tolist(), W3.tolist(), tol=1e-8):
                        # Round to half-integers and check again
                        U3_r, V3_r, W3_r = round_to_half_integers(U3, V3, W3)
                        if check_valid(U3_r.tolist(), V3_r.tolist(), W3_r.tolist(), tol=1e-8):
                            return U3_r.tolist(), V3_r.tolist(), W3_r.tolist(), new_r
                        else:
                            return U3.tolist(), V3.tolist(), W3.tolist(), new_r
                except Exception:
                    pass
            return None
        # Round to half-integers and check again
        U2_r, V2_r, W2_r = round_to_half_integers(U2, V2, W2)
        if check_valid(U2_r.tolist(), V2_r.tolist(), W2_r.tolist(), tol=1e-8):
            return U2_r.tolist(), V2_r.tolist(), W2_r.tolist(), new_r
        else:
            return U2.tolist(), V2.tolist(), W2.tolist(), new_r

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    rng = random.Random(12345)

    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass
    
    # Refinement: iteratively merge columns on the best valid decomposition
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])

        max_attempts = 1000
        attempts = 0
        target = 93
        stall_count = 0
        max_stall = 30
        
        while cur_rank > target and attempts < max_attempts and elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            merged = False
            # Try merging 1 first, then 2
            for n_merge in (1, 2):
                if cur_rank - n_merge < 1:
                    continue
                res = random_merge(cur_U, cur_V, cur_W, n_merge, rng, tol=1e-9)
                attempts += 1
                if res is not None:
                    cur_U, cur_V, cur_W, cur_rank = res
                    merged = True
                    stall_count = 0
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_result = {
                            "U": cur_U.tolist(),
                            "V": cur_V.tolist(),
                            "W": cur_W.tolist(),
                        }
                    break
                if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                    break
            if not merged:
                stall_count += 1
                if stall_count >= max_stall:
                    break
                # Perturb and try again
                if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
                    for _ in range(10):
                        res = random_merge(cur_U, cur_V, cur_W, 1, rng, tol=1e-9)
                        attempts += 1
                        if res is not None:
                            cur_U, cur_V, cur_W, cur_rank = res
                            stall_count = 0
                            if cur_rank < best_rank:
                                best_rank = cur_rank
                                best_result = {
                                    "U": cur_U.tolist(),
                                    "V": cur_V.tolist(),
                                    "W": cur_W.tolist(),
                                }
                            break
                        if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                            break
                    else:
                        break

    return best_result

import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    schoolbook_rank = n * m * p  # 112
    U_2, V_2, W_2 = _strassen_2x2x2()

    # 4x4x4 Strassen Kronecker: rank 49
    U_444 = np.zeros((16, 49))
    V_444_std = np.zeros((16, 49))
    W_444_std = np.zeros((16, 49))
    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V_444_std[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W_444_std[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])

    V_444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_444_full[j * 7 + k, :] = V_444_std[j * 4 + k, :]
    W_444_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_444_full[k * 4 + i, :] = W_444_std[k * 4 + i, :]

    # 4x4x3 schoolbook: rank 48
    U_right = np.zeros((16, 48))
    V_right = np.zeros((28, 48))
    W_right = np.zeros((28, 48))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_right[i * 4 + j, col] = 1
                V_right[j * 7 + k, col] = 1
                W_right[k * 4 + i, col] = 1
                col += 1

    U_total = np.hstack([U_444, U_right])
    V_total = np.hstack([V_444_full, V_right])
    W_total = np.hstack([W_444_full, W_right])

    recon = np.einsum('ak,bk,ck->abc', U_total, V_total, W_total)
    if np.allclose(recon, T, atol=1e-10):
        best_rank = 97
        best_result = {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        return _schoolbook(n, m, p)

    # Try to improve 4x4x3 part via greedy peeling
    if time.time() < deadline - 15.0:
        T_right = np.zeros((16, 28, 28), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    T_right[i * 4 + j, j * 7 + k, k * 4 + i] = 1

        right_result = _greedy_peel(T_right, 16, 28, 28, min(deadline - 10.0, time.time() + 50.0), 47)
        if right_result is not None:
            r_right = len(right_result["U"][0])
            U_r = np.array(right_result["U"]).T
            V_r = np.array(right_result["V"]).T
            W_r = np.array(right_result["W"]).T

            U_comb = np.hstack([U_444, U_r])
            V_comb = np.hstack([V_444_full, V_r])
            W_comb = np.hstack([W_444_full, W_r])
            recon2 = np.einsum('ak,bk,ck->abc', U_comb, V_comb, W_comb)
            if np.allclose(recon2, T, atol=1e-10):
                if 49 + r_right < best_rank:
                    best_rank = 49 + r_right
                    best_result = {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}

    return best_result


def _greedy_peel(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """Greedy rank-1 peeling."""
    R = T.copy()
    U_list, V_list, W_list = [], [], []

    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break

        best_score = 0
        best_u, best_v, best_w = None, None, None

        for restart in range(6):
            if time.time() > deadline:
                break
            if restart == 0:
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            elif restart == 1:
                cu = np.random.choice([-1, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 1], size=dim_w).astype(np.float64)
            else:
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)

            for _ in range(12):
                if time.time() > deadline:
                    break
                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)
                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)
                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)
                cu, cv, cw = new_u, new_v, new_w

            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()

        if best_score <= 0 or best_u is None:
            break

        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)

        if np.allclose(R, 0, atol=1e-10):
            break

    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        recon = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(recon, T, atol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

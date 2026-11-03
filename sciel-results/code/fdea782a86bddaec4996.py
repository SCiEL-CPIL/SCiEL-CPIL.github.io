"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _get_school_221():
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    return U, V, W

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[jp*p+k, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _build_rank50(n, m, p):
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [(0, 2, 'strassen'), (2, 4, 'strassen'), (4, 6, 'strassen'), (6, 7, 'school')]
    total_rank = 0
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            total_rank += 7 if p_type == 'strassen' else 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col_offset = 0
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            p_sub = k_end - k_start
            m_sub = j_end - j_start
            for i in range(n):
                for j_sub in range(m_sub):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            for j_sub in range(m_sub):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            col_offset += r_sub
    return U, V, W, total_rank

def _tensor_error(U, V, W, n, m, p):
    r = U.shape[1]
    err = np.zeros((n, m, m, p))
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            uc_i = uc[i*m:(i+1)*m]
            for k in range(p):
                wc_ki = wc[k*n+i]
                if wc_ki != 0:
                    vc_k = vc[k*m:(k+1)*m]
                    err[i, :, :, k] += np.outer(uc_i, vc_k) * wc_ki
    for i in range(n):
        for j in range(m):
            for k in range(p):
                err[i, j, j, k] -= 1.0
    return err

def _gradients(err, U, V, W, n, m, p):
    r = U.shape[1]
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            uc_i = uc[i*m:(i+1)*m]
            for k in range(p):
                wc_ki = wc[k*n+i]
                if wc_ki == 0:
                    continue
                vc_k = vc[k*m:(k+1)*m]
                err_ik = err[i, :, :, k]
                dU[i*m:(i+1)*m, c] += err_ik @ vc_k * wc_ki
                dV[k*m:(k+1)*m, c] += (err_ik.T @ uc_i) * wc_ki
                dW[k*n+i, c] += np.sum(err_ik * np.outer(uc_i, vc_k))
    return dU, dV, dW

def _try_reduce_rank(U0, V0, W0, target_rank, deadline, n, m, p):
    r0 = U0.shape[1]
    if target_rank >= r0:
        return None
    rng = np.random.RandomState(target_rank * 7 + 13)
    idx = rng.choice(r0, size=target_rank, replace=False)
    U = U0[:, idx].copy()
    V = V0[:, idx].copy()
    W = W0[:, idx].copy()
    U += rng.randn(*U.shape) * 0.1
    V += rng.randn(*V.shape) * 0.1
    W += rng.randn(*W.shape) * 0.1
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    lr = 0.02
    for iteration in range(5000):
        if time.time() > deadline:
            break
        err = _tensor_error(U, V, W, n, m, p)
        cur_err = np.sum(err**2)
        if cur_err < best_err:
            best_err = cur_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-14:
            break
        dU, dV, dW = _gradients(err, U, V, W, n, m, p)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        if iteration % 100 == 0:
            lr *= 0.9
    if best_err < 1e-10:
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        U_r = np.round(best_U)
        V_r = np.round(best_V)
        W_r = np.round(best_W)
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    U, V, W, known_rank = _build_rank50(n, m, p)
    if _verify(U, V, W, n, m, p):
        known_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        r = n * m * p
        U_fb = [[0] * r for _ in range(n * m)]
        V_fb = [[0] * r for _ in range(m * p)]
        W_fb = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        known_result = {"U": U_fb, "V": V_fb, "W": W_fb}
        known_rank = r
    for target_rank in range(known_rank - 1, max(known_rank - 6, 27), -1):
        if time.time() > deadline - 5:
            break
        result = _try_reduce_rank(U, V, W, target_rank, deadline - 3, n, m, p)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return known_result

# EVOLVE-BLOCK-END

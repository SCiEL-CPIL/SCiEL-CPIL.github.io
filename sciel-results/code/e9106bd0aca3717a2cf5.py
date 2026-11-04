import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_56():
    """Build the rank-56 Strassen decomposition for 2x4 * 4x8."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j, col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k, col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i, col_idx] += val

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if _time_left() < 5.0:
                break

            # Strassen 7 terms for 2x2x2 block
            # M1 = (A00+A11)(B00+B11) -> C00:1, C11:1
            add_term(
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1

            # M2 = (A10+A11)B00 -> C10:1, C11:-1
            add_term(
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1, (k0+1, 1): -1},
                col
            )
            col += 1

            # M3 = A00(B01-B11) -> C01:1, C11:1
            add_term(
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1

            # M4 = A11(B10-B00) -> C00:1, C10:1
            add_term(
                {(1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, 0): 1, (k0, 1): 1},
                col
            )
            col += 1

            # M5 = (A00+A01)B11 -> C00:-1, C01:1
            add_term(
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, 0): -1, (k0+1, 0): 1},
                col
            )
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11:1
            add_term(
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, 1): 1},
                col
            )
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00:1
            add_term(
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1},
                col
            )
            col += 1

    return U, V, W

def _build_schoolbook():
    """Build the rank-64 schoolbook decomposition."""
    n, m, p = 2, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Verify decomposition reproduces the matmul tensor exactly."""
    n, m, p = 2, 4, 8
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_recon, T_target, atol=tol)

def _try_reduce_rank(U, V, W, max_reduction=3):
    """Try to reduce rank by finding scalar-proportional duplicate columns."""
    n, m, p = 2, 4, 8
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank

    for trial in range(max_reduction):
        if _time_left() < 20:
            break

        reduced = False
        for c in range(current_rank):
            if _time_left() < 20:
                break
            for c2 in range(current_rank):
                if c2 == c:
                    continue
                # Check if column c is scalar multiple of c2 in all three matrices
                nz = np.nonzero(U[:, c2])[0]
                if len(nz) == 0:
                    continue
                idx = nz[0]
                if U[idx, c2] == 0:
                    continue
                alpha = U[idx, c] / U[idx, c2]
                if (np.allclose(U[:, c], alpha * U[:, c2], atol=1e-10) and
                    np.allclose(V[:, c], alpha * V[:, c2], atol=1e-10) and
                    np.allclose(W[:, c], alpha * W[:, c2], atol=1e-10)):
                    mask = np.ones(current_rank, dtype=bool)
                    mask[c] = False
                    best_U = U[:, mask].copy()
                    best_V = V[:, mask].copy()
                    best_W = W[:, mask].copy()
                    current_rank -= 1
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            break

    if _verify(best_U, best_V, best_W):
        return best_U, best_V, best_W, best_rank
    return U, V, W, current_rank

def construct():
    n, m, p = 2, 4, 8

    U, V, W = _build_strassen_56()

    if not _verify(U, V, W):
        U, V, W = _build_schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    U_red, V_red, W_red, rank_red = _try_reduce_rank(U, V, W, max_reduction=3)

    if rank_red < U.shape[1]:
        return {
            "U": U_red.tolist(),
            "V": V_red.tolist(),
            "W": W_red.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))

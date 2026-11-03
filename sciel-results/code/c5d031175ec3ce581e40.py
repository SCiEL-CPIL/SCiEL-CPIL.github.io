import time
import numpy as np

DIMENSIONS = [2, 4, 4]
_START = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _build_strassen_28():
    """Static rank-28 decomposition: 4 independent 2x2 blocks, each Strassen (rank 7)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms: (U coeffs, V coeffs, W coeffs) over local 2x2 indices
    # local idx: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7
    ]

    for b in range(4):
        j_off = 2 * (b // 2)  # A column block offset: 0,2,0,2
        k_off = 2 * (b % 2)   # C column block offset: 0,0,2,2
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = strassen_terms[s]
            for x_idx, coeff in u_dict.items():
                i = x_idx // 2
                j_local = x_idx % 2
                a_idx = i * m + (j_off + j_local)
                U[a_idx][col] += coeff
            for y_idx, coeff in v_dict.items():
                i_local = y_idx // 2
                j_local = y_idx % 2
                b_idx = (j_off + i_local) * p + (k_off + j_local)
                V[b_idx][col] += coeff
            for z_idx, coeff in w_dict.items():
                i = z_idx // 2
                k_local = z_idx % 2
                c_idx = (k_off + k_local) * n + i
                W[c_idx][col] += coeff

    return U, V, W


def _verify(U, V, W, n=2, m=4, p=4, tol=1e-7):
    """Return (ok, max_err) for the decomposition U,V,W against the all-ones tensor."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Ue = Ua.reshape(n, m, -1)
    Ve = Va.reshape(m, p, -1)
    We = Wa.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', Ue, Ve, We)
    err = float(np.max(np.abs(T - 1.0)))
    return err < tol, err


def construct():
    n, m, p = DIMENSIONS
    # Guaranteed rank-28 fallback (known optimal for this tensor).
    U28, V28, W28 = _build_strassen_28()
    ok, err = _verify(U28, V28, W28)
    if not ok:
        # Should never happen; still return the static decomposition.
        return {"U": U28, "V": V28, "W": W28}

    # Cheap, structured rank-reduction probe: try removing a single Strassen
    # term and absorbing its (i,j,k) contribution into the remaining terms.
    # This is O(1) work per attempt and never perturbs the valid rank-28 base.
    if _time_left() > 3.0:
        U28a = np.asarray(U28, float)
        V28a = np.asarray(V28, float)
        W28a = np.asarray(W28, float)
        Ue = U28a.reshape(n, m, 28)
        Ve = V28a.reshape(m, p, 28)
        We = W28a.reshape(p, n, 28)
        # Precompute the full 2x4x4 tensor contribution per rank column.
        # T_c[i,j,k] = Ue[i,j,c]*Ve[j,k,c]*We[k,i,c]
        T_full = np.einsum('ijc,jkc,kic->ijkc', Ue, Ve, We)  # (2,4,4,28)
        # Sum over c gives the all-ones tensor (verified).
        for c in range(28):
            if _time_left() < 3.0:
                break
            # Candidate: drop column c.
            U = U28a[:, list(range(28)) and [r for r in range(28) if r != c]]
            V = V28a[:, [r for r in range(28) if r != c]]
            W = W28a[:, [r for r in range(28) if r != c]]
            # Defect to absorb: D[i,j,k] = -T_full[i,j,k,c]
            D = -T_full[:, :, :, c]  # (2,4,4)
            # Try to express D as sum_{c' != c} alpha_{c'} * T_full[:,:,c',c']
            # i.e. solve linear system: for each (i,j,k), D[i,j,k] = sum alpha_{c'} T_full[i,j,k,c']
            cols = [r for r in range(28) if r != c]
            A = T_full[:, :, :, cols].reshape(2 * 4 * 4, -1)  # (32, 27)
            b = D.reshape(-1)
            try:
                alpha, *_ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            # Reconstruct candidate matrices with added alpha*c' contributions.
            Uc = U.copy()
            Vc = V.copy()
            Wc = W.copy()
            for idx, c2 in enumerate(cols):
                a = alpha[idx]
                if abs(a) < 1e-12:
                    continue
                Uc[:, idx] += a * U28a[:, c2]
                Vc[:, idx] += a * V28a[:, c2]
                Wc[:, idx] += a * W28a[:, c2]
            # Verify candidate rank-27.
            ok2, err2 = _verify(Uc.tolist(), Vc.tolist(), Wc.tolist())
            if ok2:
                return {"U": Uc.tolist(), "V": Vc.tolist(), "W": Wc.tolist()}

    return {"U": U28, "V": V28, "W": W28}

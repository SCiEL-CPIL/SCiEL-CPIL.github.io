"""Matrix multiplication 2,2,8: rank-minimal exact tensor decomposition.

Keeps the verified rank-28 Strassen construction as a guaranteed fallback and
adds a bounded, deadline-guarded search for a rank-27 exact decomposition.
"""
DIMENSIONS = [2, 2, 8]
import time

# EVOLVE-BLOCK-START
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0


def _strassen_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for block in range(4):
        col_offset = block * 2
        v_00 = col_offset
        v_01 = col_offset + 1
        v_10 = 8 + col_offset
        v_11 = 8 + col_offset + 1
        w_00 = 2 * col_offset
        w_01 = 2 * (col_offset + 1)
        w_10 = 2 * col_offset + 1
        w_11 = 2 * (col_offset + 1) + 1
        u_00, u_01, u_10, u_11 = 0, 1, 2, 3
        base_col = block * 7

        col = base_col
        U[u_00][col] += 1; U[u_11][col] += 1
        V[v_00][col] += 1; V[v_11][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1

        col = base_col + 1
        U[u_10][col] += 1; U[u_11][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1; W[w_11][col] -= 1

        col = base_col + 2
        U[u_00][col] += 1
        V[v_01][col] += 1; V[v_11][col] -= 1
        W[w_01][col] += 1; W[w_11][col] += 1

        col = base_col + 3
        U[u_11][col] += 1
        V[v_10][col] += 1; V[v_00][col] -= 1
        W[w_00][col] += 1; W[w_10][col] += 1

        col = base_col + 4
        U[u_00][col] += 1; U[u_01][col] += 1
        V[v_11][col] += 1
        W[w_00][col] -= 1; W[w_01][col] += 1

        col = base_col + 5
        U[u_10][col] += 1; U[u_00][col] -= 1
        V[v_00][col] += 1; V[v_01][col] += 1
        W[w_11][col] += 1

        col = base_col + 6
        U[u_01][col] += 1; U[u_11][col] -= 1
        V[v_10][col] += 1; V[v_11][col] += 1
        W[w_00][col] += 1

    return U, V, W


def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(r):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(s) > 1e-6:
                    return False
    return True


def _try_rank27():
    """Bounded search for a rank-27 exact decomposition.

    Structure: 4 independent 2x2 blocks. Three blocks use the 7-term Strassen
    basis (21 terms) and the fourth block is solved for its 6 term vectors
    directly (the 2x2x2 tensor has rank 7, so a 6-term exact decomposition of
    one full block is impossible -- this search therefore only succeeds if the
    tensor admits a genuine rank-27 structure, which it is expected not to).
    Returns (U,V,W) on success or None.
    """
    n, m, p = DIMENSIONS
    U, V, W = _strassen_rank28()
    r28 = 28

    # Build the 2x2x2 tensor T (a,b) -> c for a 2x2 * 2x2 product.
    T = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i * 2 + j, j * 2 + k, k * 2 + i] = 1.0

    # Strassen 7-term basis for a 2x2 block (u, v, w coefficient vectors).
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))   # P1
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))  # P2
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))  # P3
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))  # P4
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))  # P5
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))  # P6
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))  # P7

    # Fixed 21 terms from the first three blocks (columns 0..20 of the rank-28).
    def build_fixed(base_col, block):
        Uf = np.zeros((4, 21))
        Vf = np.zeros((4, 21))
        Wf = np.zeros((4, 21))
        co = block * 2
        vmap = {0: co, 1: co + 1, 2: 8 + co, 3: 8 + co + 1}
        wmap = {0: 2 * co, 1: 2 * (co + 1), 2: 2 * co + 1, 3: 2 * (co + 1) + 1}
        for t_idx, (uu, vv, ww) in enumerate(terms):
            c = base_col + t_idx
            for a in range(4):
                Uf[a, t_idx] = uu[a]
            for b in range(4):
                Vf[b, t_idx] = vv[b]
            for cix in range(4):
                Wf[cix, t_idx] = ww[cix]
        return Uf, Vf, Wf

    Uf, Vf, Wf = build_fixed(0, 0)  # block 0 as the representative 2x2 block

    # Residual tensor for the fourth block: R = T - sum_{fixed} u*v*w
    R = T.copy()
    for t in range(21):
        u = Uf[:, t]
        v = Vf[:, t]
        w = Wf[:, t]
        R -= np.einsum('a,b,c->abc', u, v, w)

    # Solve for 6 vectors x_t (a), y_t (b), z_t (c) such that
    #   sum_t x_t y_t z_t = R.
    # Linearize: treat y,z fixed, solve x linearly, etc. (multilinear).
    # We use a simple alternating scheme; if it converges to exact, success.
    rng = np.random.default_rng(12345)
    best = None
    for attempt in range(40):
        X = rng.normal(size=(4, 6))
        Y = rng.normal(size=(4, 6))
        Z = rng.normal(size=(4, 6))
        for it in range(200):
            if time.time() > _DEADLINE:
                return None
            # solve X given Y,Z:  R[a,b,c] = sum_t X[a,t] Y[b,t] Z[c,t]
            # For each a:  sum_t X[a,t] (Y[:,t]*Z[:,t])^T reshaped = R[a,:,:]
            M = (Y * Z).T  # shape (6, 4, 4) -> flatten to (6,16)
            Mf = M.reshape(6, 16)
            for a in range(4):
                rhs = R[a].reshape(16)
                sol, *_ = np.linalg.lstsq(Mf, rhs, rcond=None)
                X[a] = sol
            for b in range(4):
                M = (X * Z).T.reshape(6, 16)
                rhs = R[:, b, :].reshape(16)
                sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                Y[b] = sol
            for c in range(4):
                M = (X * Y).T.reshape(6, 16)
                rhs = R[:, :, c].reshape(16)
                sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                Z[c] = sol
            # check
            S = np.zeros((4, 4, 4))
            for t in range(6):
                S += np.einsum('a,b,c->abc', X[:, t], Y[:, t], Z[:, t])
            err = np.max(np.abs(S - R))
            if err < 1e-8:
                best = (X, Y, Z)
                break
        if best is not None:
            break

    if best is None:
        return None

    X, Y, Z = best
    # Assemble rank-27: 21 fixed terms (block0 basis) + 6 solved terms.
    # NOTE: this only covers one block's worth of structure; full 2x2x8 needs
    # all four blocks. Since a single block cannot be done in 6 terms, this is
    # expected to fail; kept as a bounded probe.
    U27 = np.zeros((n * m, 27))
    V27 = np.zeros((m * p, 27))
    W27 = np.zeros((p * n, 27))
    # Place the 21 fixed terms for block 0
    for t in range(21):
        U27[0:4, t] = Uf[:, t]
        co = 0
        V27[co:co + 2, t] = Vf[0:2, t]
        V27[8 + co:8 + co + 2, t] = Vf[2:4, t]
        W27[0:4, t] = Wf[:, t]
    # Place 6 solved terms for block 0 as well (redundant) -- skip; return None
    # because we cannot cover blocks 1,2,3 without more terms.
    return None


def construct():
    n, m, p = DIMENSIONS
    # Guaranteed rank-28 fallback.
    U, V, W = _strassen_rank28()
    result = {"U": U, "V": V, "W": W}

    # Bounded attempt at rank 27 (deadline-guarded).
    if time.time() < _DEADLINE - 10.0:
        try:
            r27 = _try_rank27()
            if r27 is not None:
                U27, V27, W27 = r27
                if _verify(U27, V27, W27, n, m, p):
                    result = {"U": U27, "V": V27, "W": W27}
        except Exception:
            pass

    return result
# EVOLVE-BLOCK-END

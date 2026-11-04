import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _build_strassen_blocks(n, m, p):
    """Return list of (U_block, V_block, W_block) triples, each a 2x2x2 Strassen
    block (rank 7), covering the full 2x4x6 multiplication. 6 blocks total."""
    blocks = []
    positions = []
    for j in (0, 2):
        for k in (0, 2, 4):
            positions.append((0, j, k))

    for (i0, j0, k0) in positions:
        U = np.zeros((n, m, 7), dtype=float)
        V = np.zeros((m, p, 7), dtype=float)
        W = np.zeros((p, n, 7), dtype=float)

        a = lambda di: i0 + di
        b = lambda dj: j0 + dj
        c = lambda dk: k0 + dk

        # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
        U[a(0), b(0), 0] += 1; U[a(1), b(1), 0] += 1
        V[b(0), c(0), 0] += 1; V[b(1), c(1), 0] += 1
        W[c(0), a(0), 0] += 1; W[c(1), a(1), 0] += 1

        # M2: (A10+A11)B00 -> C10+1, C11-1
        U[a(1), b(0), 1] += 1; U[a(1), b(1), 1] += 1
        V[b(0), c(0), 1] += 1
        W[c(0), a(1), 1] += 1; W[c(1), a(1), 1] -= 1

        # M3: A00(B01-B11) -> C01+1, C11+1
        U[a(0), b(0), 2] += 1
        V[b(0), c(1), 2] += 1; V[b(1), c(1), 2] -= 1
        W[c(1), a(0), 2] += 1; W[c(1), a(1), 2] += 1

        # M4: A11(B10-B00) -> C00+1, C10+1
        U[a(1), b(1), 3] += 1
        V[b(1), c(0), 3] += 1; V[b(0), c(0), 3] -= 1
        W[c(0), a(0), 3] += 1; W[c(0), a(1), 3] += 1

        # M5: (A00+A01)B11 -> C00-1, C01+1
        U[a(0), b(0), 4] += 1; U[a(0), b(1), 4] += 1
        V[b(1), c(1), 4] += 1
        W[c(0), a(0), 4] -= 1; W[c(1), a(0), 4] += 1

        # M6: (A10-A00)(B00+B01) -> C11+1
        U[a(1), b(0), 5] += 1; U[a(0), b(0), 5] -= 1
        V[b(0), c(0), 5] += 1; V[b(0), c(1), 5] += 1
        W[c(1), a(1), 5] += 1

        # M7: (A01-A11)(B10+B11) -> C00+1
        U[a(0), b(1), 6] += 1; U[a(1), b(1), 6] -= 1
        V[b(1), c(0), 6] += 1; V[b(1), c(1), 6] += 1
        W[c(0), a(0), 6] += 1

        blocks.append((U, V, W))
    return blocks


def _decomp_to_matrices(blocks, n, m, p, rank):
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for r, (Ub, Vb, Wb) in enumerate(blocks):
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = Ub[i, j, r]
        for j in range(m):
            for k in range(p):
                V[j * p + k, r] = Vb[j, k, r]
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = Wb[k, i, r]
    return U, V, W


def _check_decomp(U, V, W, n, m, p, rank, tol=1e-6):
    """True iff sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= tol))


def _round_half(U, V, W):
    U = np.round(2.0 * U) / 2.0
    V = np.round(2.0 * V) / 2.0
    W = np.round(2.0 * W) / 2.0
    return U, V, W


def _greedy_reduce(n, m, p, deadline):
    """Try to remove rank-1 terms that are linear combinations of the rest."""
    blocks = _build_strassen_blocks(n, m, p)
    R0 = 7 * len(blocks)
    U, V, W = _decomp_to_matrices(blocks, n, m, p, R0)

    removed = set()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(R0):
            if r in removed:
                continue
            if time.time() > deadline:
                break
            keep = [rr for rr in range(R0) if rr != r and rr not in removed]
            if not keep:
                break
            u = U[:, r].reshape(-1)
            v = V[:, r].reshape(-1)
            w = W[:, r].reshape(-1)
            Uo = U[:, keep].reshape(-1, len(keep))
            Vo = V[:, keep].reshape(-1, len(keep))
            Wo = W[:, keep].reshape(-1, len(keep))
            # solve for alpha,beta,gamma with (u,v,w) = sum_s alpha_s (Uo_s,Vo_s,Wo_s)
            A = np.vstack([Uo, Vo, Wo])
            bvec = np.concatenate([u, v, w])
            try:
                sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
            except np.linalg.LinAlgError:
                continue
            residual = np.linalg.norm(A @ sol - bvec)
            scale = np.linalg.norm(bvec) + 1e-12
            if residual <= 1e-7 * max(1.0, scale):
                removed.add(r)
                improved = True

    if not removed:
        return None
    keep = [rr for rr in range(R0) if rr not in removed]
    U2 = U[:, keep]
    V2 = V[:, keep]
    W2 = W[:, keep]
    if _check_decomp(U2, V2, W2, n, m, p, len(keep)):
        return U2, V2, W2
    return None


def _random_search(n, m, p, deadline):
    """Random half-integer init + gradient descent on residual objective."""
    rng = np.random.default_rng(12345)
    R0 = 42
    best = None
    best_rank = R0
    U0, V0, W0 = _decomp_to_matrices(_build_strassen_blocks(n, m, p), n, m, p, R0)

    for rank in range(R0 - 1, 35, -1):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(8):
            if time.time() > deadline:
                break
            # random half-integer init
            U = (rng.integers(-4, 5, size=(n * m, rank)) / 2.0)
            V = (rng.integers(-4, 5, size=(m * p, rank)) / 2.0)
            W = (rng.integers(-4, 5, size=(p * n, rank)) / 2.0)

            lr = 0.02
            for it in range(600):
                if time.time() > deadline:
                    break
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                resid = T - 1.0
                if np.linalg.norm(resid, ord=np.inf) < 1e-7:
                    break
                # gradients
                dU = np.einsum('ijk,jkr,ijr->ijr', resid, V3, W3)
                dV = np.einsum('ijk,ijr,kir->jkr', resid, U3, W3)
                dW = np.einsum('ijk,ijr,jkr->kir', resid, U3, V3)
                U -= lr * dU.reshape(n * m, rank)
                V -= lr * dV.reshape(m * p, rank)
                W -= lr * dW.reshape(p * n, rank)

            U, V, W = _round_half(U, V, W)
            if _check_decomp(U, V, W, n, m, p, rank, tol=1e-6):
                best = (U, V, W)
                best_rank = rank
                found = True
                break
        if found:
            continue
        # no rank found at this level; stop
        break

    if best is None:
        return None
    return best


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    R0 = 42

    # Guaranteed rank-42 fallback
    blocks = _build_strassen_blocks(n, m, p)
    U, V, W = _decomp_to_matrices(blocks, n, m, p, R0)
    if not _check_decomp(U, V, W, n, m, p, R0):
        # schoolbook fallback (should never trigger)
        R = n * m * p
        U = np.zeros((n * m, R)); V = np.zeros((m * p, R)); W = np.zeros((p * n, R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = (U, V, W)

    # (a) greedy linear-dependence elimination
    try:
        red = _greedy_reduce(n, m, p, deadline)
        if red is not None and red[0].shape[1] < best[0].shape[1]:
            best = red
    except Exception:
        pass

    # (b) random half-integer + gradient descent search
    try:
        rand = _random_search(n, m, p, deadline)
        if rand is not None and rand[0].shape[1] < best[0].shape[1]:
            best = rand
    except Exception:
        pass

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

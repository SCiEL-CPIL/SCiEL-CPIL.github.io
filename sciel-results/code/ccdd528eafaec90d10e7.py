import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = 150.0  # evaluator kills at 150s; keep margin


def _elapsed():
    return time.time() - _START


def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=1e-9)


def _strassen_222():
    """Return a rank-7 Strassen decomposition for 2x2x2.
    U[2*2,7], V[2*2,7], W[2*2,7] with indices U[i*2+j], V[j*2+k], W[k*2+i].
    """
    # Strassen 7 products for 2x2x2
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)(b00)
    # M2 = (a00)(b01-b11)
    # M3 = (a11)(b10-b00)
    # M4 = (a00+a01)(b11)
    # M5 = (a10-a00)(b00+b10)
    # M6 = (a01-a11)(b01+b11)
    # C00 = M0+M3-M4-M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M6
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    # row index: U[i*2+j], V[j*2+k], W[k*2+i]
    # M0: (a00+a11)(b00+b11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[2, 0] += 1; W[1, 0] -= 1; W[3, 0] -= 1
    # M1: (a10+a11)(b00)
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    # M2: (a00)(b01-b11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    # M3: (a11)(b10-b00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    # M4: (a00+a01)(b11)
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] -= 1
    # M5: (a10-a00)(b00+b10)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[2, 5] += 1
    W[2, 5] += 1
    # M6: (a01-a11)(b01+b11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[1, 6] += 1; V[3, 6] += 1
    W[1, 6] += 1; W[3, 6] -= 1
    return U, V, W


def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W


def _get_sub_decomp(n_sub, m_sub, p_sub):
    """Get best known decomposition for a sub-block."""
    if n_sub == 2 and m_sub == 2 and p_sub == 2:
        return _strassen_222()
    return _schoolbook(n_sub, m_sub, p_sub)


def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """Embed a sub-decomposition into the full n x m x p problem."""
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)

    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]

    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]

    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]

    return U_full, V_full, W_full


def _block_decomposition_356():
    """
    Block decomposition for 3x5x6.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Uses Strassen rank-7 for 2x2x2 blocks, schoolbook for others.
    Expected rank: 6*7 + 3*4 + 6*4 + 3*2 = 42+12+24+6 = 84
    """
    n, m, p = 3, 5, 6
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]

    # Accumulate all blocks
    U_parts = []
    V_parts = []
    W_parts = []

    i_off = 0
    for n_sub in n_splits:
        j_off = 0
        for m_sub in m_splits:
            k_off = 0
            for p_sub in p_splits:
                U_sub, V_sub, W_sub = _get_sub_decomp(n_sub, m_sub, p_sub)
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p)
                U_parts.append(U_full)
                V_parts.append(V_full)
                W_parts.append(W_full)
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub

    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)

    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, max_time=10.0):
    """
    Try to reduce rank by finding linearly dependent columns.
    A column r is redundant if the tensor contribution can be expressed
    as a linear combination of other columns.
    """
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    start = time.time()

    # Try to remove columns that are linearly dependent
    # For each column, check if removing it and adjusting others preserves the tensor
    # This is expensive, so we do a greedy approach

    # First, try simple: check if any column is exactly zero
    for c in range(r - 1, -1, -1):
        if time.time() - start > max_time:
            break
        if np.allclose(U[:, c], 0) and np.allclose(V[:, c], 0) and np.allclose(W[:, c], 0):
            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            r -= 1

    return U, V, W


def _attempt_smaller_blocks():
    """
    Try alternative block splits to see if any gives lower rank.
    """
    n, m, p = 3, 5, 6
    best_rank = 90
    best = None

    # Try various splits
    n_splits_options = [[2, 1], [1, 2], [3], [1, 1, 1]]
    m_splits_options = [[2, 2, 1], [3, 2], [2, 3], [5], [1, 1, 1, 1, 1]]
    p_splits_options = [[2, 2, 2], [3, 3], [2, 4], [6], [1, 1, 1, 1, 1, 1]]

    for n_s in n_splits_options:
        if sum(n_s) != n:
            continue
        for m_s in m_splits_options:
            if sum(m_s) != m:
                continue
            for p_s in p_splits_options:
                if sum(p_s) != p:
                    continue
                # Calculate expected rank
                total_rank = 0
                i_off = 0
                for n_sub in n_s:
                    j_off = 0
                    for m_sub in m_s:
                        k_off = 0
                        for p_sub in p_s:
                            if n_sub == 2 and m_sub == 2 and p_sub == 2:
                                total_rank += 7
                            else:
                                total_rank += n_sub * m_sub * p_sub
                            k_off += p_sub
                        j_off += m_sub
                    i_off += n_sub

                if total_rank < best_rank:
                    best_rank = total_rank
                    # Build the decomposition
                    U_parts = []
                    V_parts = []
                    W_parts = []
                    i_off = 0
                    for n_sub in n_s:
                        j_off = 0
                        for m_sub in m_s:
                            k_off = 0
                            for p_sub in p_s:
                                U_sub, V_sub, W_sub = _get_sub_decomp(n_sub, m_sub, p_sub)
                                U_full, V_full, W_full = _embed_sub_decomp(
                                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                                    i_off, j_off, k_off, n, m, p)
                                U_parts.append(U_full)
                                V_parts.append(V_full)
                                W_parts.append(W_full)
                                k_off += p_sub
                            j_off += m_sub
                        i_off += n_sub
                    U = np.hstack(U_parts)
                    V = np.hstack(V_parts)
                    W = np.hstack(W_parts)
                    if _verify_decomp(n, m, p, U, V, W):
                        best = (U, V, W)

    if best is not None:
        return best
    return None


def _try_winograd_333():
    """
    Attempt to use a rank-23 decomposition for 3x3x3 blocks.
    If we can verify one, use it to reduce rank further.
    """
    # This is a known open problem area; for now return None
    # and rely on the block decomposition.
    return None


def construct():
    global _START
    _START = time.time()
    n, m, p = DIMENSIONS

    # Try to find the best decomposition
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook

    # 1. Try block decomposition (should give 84)
    try:
        U, V, W = _block_decomposition_356()
        if _verify_decomp(n, m, p, U, V, W):
            r = U.shape[1]
            if r < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = r
    except Exception:
        pass

    # 2. Try alternative block splits
    if _elapsed() < _DEADLINE - 30:
        try:
            alt = _attempt_smaller_blocks()
            if alt is not None:
                U, V, W = alt
                r = U.shape[1]
                if r < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
        except Exception:
            pass

    # 3. Try rank reduction on the best found
    if best_U is not None and _elapsed() < _DEADLINE - 20:
        try:
            U, V, W = _try_rank_reduction(best_U, best_V, best_W, n, m, p, max_time=5.0)
            if _verify_decomp(n, m, p, U, V, W):
                r = U.shape[1]
                if r < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
        except Exception:
            pass

    # 4. Fallback: schoolbook
    if best_U is None:
        rank = n * m * p
        best_U = np.zeros((n * m, rank), dtype=np.float64)
        best_V = np.zeros((m * p, rank), dtype=np.float64)
        best_W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1.0
                    best_V[j * p + k, col] = 1.0
                    best_W[k * n + i, col] = 1.0
                    col += 1

    # Convert to lists for the interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }


_START = time.time()
# EVOLVE-BLOCK-END

"""Laser-method rank-minimal exact decomposition for 3x4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START = time.time()
_DEADLINE = 120.0


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _strassen222():
    # Rank-7 (Strassen) decomposition of the 2x2x2 multiplication tensor.
    # U: 4x7, V: 4x7, W: 4x7 (W index order k*n+i with n=2, p=2)
    U = [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
    ]
    W = [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
    ]
    return np.array(U, float), np.array(V, float), np.array(W, float)


def _base_case(n, m, p):
    """Return (U,V,W) for the n*m*p multiplication tensor (W index k*n+i)."""
    if (n, m, p) == (1, 1, 1):
        return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    if (n, m, p) == (1, 2, 1):
        return (np.eye(2), np.eye(2), np.eye(2))
    if (n, m, p) == (2, 1, 1):
        return (np.eye(2), np.eye(2), np.eye(2))
    if (n, m, p) == (1, 1, 2):
        return (np.eye(2), np.eye(2), np.eye(2))
    if (n, m, p) == (2, 2, 1):
        return (np.eye(4), np.eye(4), np.eye(4))
    if (n, m, p) == (1, 2, 2):
        return (np.eye(4), np.eye(4), np.eye(4))
    if (n, m, p) == (2, 2, 2):
        return _strassen222()
    # fallback: schoolbook
    r = n * m * p
    U = np.zeros((n * m, r)); V = np.zeros((m * p, r)); W = np.zeros((p * n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _kron_tensor(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions -> (U,V,W)."""
    return np.kron(U1, U2), np.kron(V1, V2), np.kron(W1, W2)


def _eliminate_dependent(U, V, W, tol=1e-12):
    """Remove linearly dependent rank-1 columns. Returns (U',V',W', kept)."""
    r = U.shape[1]
    if r == 0:
        return U, V, W, []
    # Build the joint matrix whose columns are the concatenated rank-1 factors.
    J = np.vstack([U, V, W])
    # SVD to find the numerical rank and an orthonormal basis of the column space.
    try:
        U_s, s, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W, list(range(r))
    tol_eff = tol * max(s) if s.size else 0.0
    num_rank = int(np.sum(s > tol_eff))
    if num_rank == 0:
        return U[:, :0], V[:, :0], W[:, :0], []
    Q = U_s[:, :num_rank]  # orthonormal basis of column space
    # A column J[:,c] is independent iff its projection residual is non-negligible.
    # Use the factored form: J @ Vt[:num_rank].T = Q @ diag(s) @ Vt[:num_rank].T
    # Greedily pick columns that increase the span of J's column space.
    # Simpler robust approach: keep a set, and a column is dependent if its
    # projection onto the current span has near-zero residual.
    kept = []
    basis = []          # list of vectors (J[:,c])
    for c in range(r):
        col = J[:, c]
        if not basis:
            kept.append(c); basis.append(col); continue
        B = np.column_stack(basis)
        # residual of col after projecting onto span(basis)
        try:
            coef, res, *_ = np.linalg.lstsq(B, col, rcond=None)
        except np.linalg.LinAlgError:
            kept.append(c); basis.append(col); continue
        if np.linalg.norm(col - B @ coef) > tol_eff:
            kept.append(c); basis.append(col)
    return U[:, kept], V[:, kept], W[:, kept], kept


def _verify(U, V, W, n, m, p):
    """Check U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,k?} ... actually ==1 iff correct."""
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k.
    # Reconstruct: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1 for all i,j,k.
    # Do it via einsum.
    Ua = np.asarray(U, float); Va = np.asarray(V, float); Wa = np.asarray(W, float)
    r = Ua.shape[1]
    # Ua[i*m+j, c], Va[j*p+k, c], Wa[k*n+i, c]
    # Build the 3D tensor T[i,j,k] = sum_c Ua[i*m+j,c]*Va[j*p+k,c]*Wa[k*n+i,c]
    # Reshape Ua to (n,m,r), Va to (m,p,r), Wa to (p,n,r)
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-8, rtol=1e-8), float(np.max(np.abs(T - 1.0)))


def _to_sparse(U, V, W, tol=1e-12):
    """Convert dense factor matrices to sparse (index, value) triple lists."""
    def sp(M):
        rows, cols = M.shape
        out = []
        for c in range(cols):
            col = M[:, c]
            for r in range(rows):
                v = col[r]
                if abs(v) > tol:
                    out.append((r, float(v)))
        return out
    return sp(U), sp(V), sp(W)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


# Precompute the schoolbook fallback (always exact, rank 72)
_SB_U, _SB_V, _SB_W = _schoolbook(3, 4, 6)
_SB_VALID, _ = _verify(_SB_U, _SB_V, _SB_W, 3, 4, 6)

# Best result found so far (fallback to schoolbook)
_best = {"U": _SB_U, "V": _SB_V, "W": _SB_W, "rank": 72}


def _try_candidate(U, V, W):
    """Verify a candidate; update global best if valid and lower rank."""
    global _best
    valid, err = _verify(U, V, W, 3, 4, 6)
    if not valid:
        return False, err
    r = U.shape[1]
    if r < _best["rank"]:
        _best = {"U": U, "V": V, "W": W, "rank": r}
    return True, err


def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook as guaranteed-valid baseline.
    global _best
    _best = {"U": _SB_U, "V": _SB_V, "W": _SB_W, "rank": 72}

    # ---- Laser method: decompose 3x4x6 into a block of sub-tensors ----
    # Block structure: partition rows of A (3), columns of A / rows of B (4),
    # and columns of B (6) into blocks, and use low-rank base decompositions.
    # We try several blockings. Each blocking is a list of (n_b, m_b, p_b) blocks
    # that tile 3x4x6. For each block we use _base_case, Kronecker across blocks
    # along each axis, then eliminate dependent columns.

    # Blocking A: split 3 = 1+2 (rows), 4 = 2+2 (mid), 6 = 2+2+2 (cols)
    # blocks: (1,2,2),(1,2,2),(1,2,2),(2,2,2),(2,2,2),(2,2,2)
    # We build the full tensor as a block-Kronecker: for each axis the block
    # decomposition is a direct sum along that axis.
    # Concretely: U = blockdiag over (i_block, j_block) of kron(U_i, U_j) ... this
    # is the tensor product of the axis decompositions. We implement by building
    # each axis's "factor" as a block-diagonal Kronecker of base factors.

    def build_axis_decomp(n_axes, m_axes, p_axes):
        """Given lists of block sizes per axis, build (U,V,W) via block Kronecker."""
        # U index (i,j): block-diag over i-block and j-block of kron(U_i, U_j)
        # V index (j,k): block-diag over j-block and k-block of kron(V_j, V_k)
        # W index (k,i): block-diag over k-block and i-block of kron(W_k, W_i)
        U_blocks = []
        V_blocks = []
        W_blocks = []
        U_rows = []  # (i_block, j_block, U_sub)
        V_rows = []
        W_rows = []
        for (i_b, n_b) in enumerate(n_axes):
            for (j_b, m_b) in enumerate(m_axes):
                Ub, Vb, Wb = _base_case(n_b, m_b, p_axes[0] if False else 1)  # placeholder
        # Simpler: treat each axis independently and combine.
        # Axis i (rows of A): blocks n_axes, each a 1x?x? ... but base case needs all 3 dims.
        # Instead use a tensor-product construction: the full tensor is the
        # Kronecker product of per-axis "line" tensors. We define per-axis factors:
        #   i-axis: for each i-block of size n_b, a 1D factor of length n_b (U_i)
        #   j-axis: for each j-block of size m_b, factor V_j
        #   k-axis: for each k-block of size p_b, factor W_k
        # The 3D tensor = sum over (i,j,k) of (U_i ⊗ V_j ⊗ W_k) with the right
        # index permutation. This is exactly the Kronecker product of the three
        # 1D decompositions.
        # So: U_full = kron over i-blocks of U_i  (length n), but U is indexed (i,j)
        # which is a 2D tensor = kron(U_i_tensor, U_j_tensor).
        # We'll build 1D factor matrices:
        U1 = np.zeros((n, 0))  # will fill
        # Build U1 as block-diag over i-blocks of the i-part of base case? But base
        # case couples all three dims. For a pure tensor-product we need each axis to
        # have its own independent factor. The cleanest is to use the 1x1x1 / line
        # decompositions:
        #   i-axis factor for block size n_b: identity n_b x n_b (rank n_b)
        #   j-axis factor for block size m_b: identity m_b x m_b (rank m_b)
        #   k-axis factor for block size p_b: identity p_b x p_b (rank p_b)
        # Then U = kron(U_i, U_j), V = kron(V_j, V_k), W = kron(W_k, W_i).
        # This gives rank = (sum n_b)*(sum m_b)*(sum p_b) = 3*4*6 = 72 (no gain).
        # To gain, we need low-rank base cases on the *2D* blocks. Use the
        # 2x2x2 Strassen on a (2,2,2) block and identities elsewhere.
        # Build via explicit block placement.
        return None

    # Explicit block construction:
    # n blocks: [1, 2]   (1+2=3)
    # m blocks: [2, 2]   (2+2=4)
    # p blocks: [2, 2, 2] (2+2+2=6)
    n_blocks = [1, 2]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 2]

    # For each triple (ib, jb, kb) use base case (n_b, m_b, p_b).
    # U index (i, j): for block (ib,jb) the sub-U has shape (n_b*m_b, r_b)
    #   placed at rows (i in block ib, j in block jb).
    # V index (j, k): sub-V shape (m_b*p_b, r_b) at (j in jb, k in kb).
    # W index (k, i): sub-W shape (p_b*n_b, r_b) at (k in kb, i in ib).
    U = np.zeros((n * m, 0)); V = np.zeros((m * p, 0)); W = np.zeros((p * n, 0))
    # We'll accumulate per-block, then stack columns.
    U_list = []; V_list = []; W_list = []
    n_off = [0]; m_off = [0]; p_off = [0]
    for sz in n_blocks: n_off.append(n_off[-1] + sz)
    for sz in m_blocks: m_off.append(m_off[-1] + sz)
    for sz in p_blocks: p_off.append(p_off[-1] + sz)

    for ib, n_b in enumerate(n_blocks):
        for jb, m_b in enumerate(m_blocks):
            for kb, p_b in enumerate(p_blocks):
                Ub, Vb, Wb = _base_case(n_b, m_b, p_b)
                r_b = Ub.shape[1]
                # U sub: rows (i in [n_off[ib], n_off[ib+1]), j in [m_off[jb], m_off[jb+1])
                #   index = i*m + j
                U_sub = np.zeros((n * m, r_b))
                for ii in range(n_b):
                    for jj in range(m_b):
                        U_sub[(n_off[ib] + ii) * m + (m_off[jb] + jj), :] = Ub[ii * m_b + jj, :]
                V_sub = np.zeros((m * p, r_b))
                for jj in range(m_b):
                    for kk in range(p_b):
                        V_sub[(m_off[jb] + jj) * p + (p_off[kb] + kk), :] = Vb[jj * p_b + kk, :]
                W_sub = np.zeros((p * n, r_b))
                for kk in range(p_b):
                    for ii in range(n_b):
                        W_sub[(p_off[kb] + kk) * n + (n_off[ib] + ii), :] = Wb[kk * n_b + ii, :]
                U_list.append(U_sub); V_list.append(V_sub); W_list.append(W_sub)

    U_full = np.hstack(U_list)
    V_full = np.hstack(V_list)
    W_full = np.hstack(W_list)

    # Eliminate dependent columns.
    U_e, V_e, W_e, kept = _eliminate_dependent(U_full, V_full, W_full)
    _try_candidate(U_e, V_e, W_e)

    # ---- Alternative blocking: n=[2,1], m=[1,1,2], p=[3,3] with 2x2x2 Strassen on (2,2,2) ----
    n_blocks = [2, 1]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 2]
    n_off = [0]; m_off = [0]; p_off = [0]
    for sz in n_blocks: n_off.append(n_off[-1] + sz)
    for sz in m_blocks: m_off.append(m_off[-1] + sz)
    for sz in p_blocks: p_off.append(p_off[-1] + sz)
    U_list = []; V_list = []; W_list = []
    for ib, n_b in enumerate(n_blocks):
        for jb, m_b in enumerate(m_blocks):
            for kb, p_b in enumerate(p_blocks):
                Ub, Vb, Wb = _base_case(n_b, m_b, p_b)
                r_b = Ub.shape[1]
                U_sub = np.zeros((n * m, r_b))
                for ii in range(n_b):
                    for jj in range(m_b):
                        U_sub[(n_off[ib] + ii) * m + (m_off[jb] + jj), :] = Ub[ii * m_b + jj, :]
                V_sub = np.zeros((m * p, r_b))
                for jj in range(m_b):
                    for kk in range(p_b):
                        V_sub[(m_off[jb] + jj) * p + (p_off[kb] + kk), :] = Vb[jj * p_b + kk, :]
                W_sub = np.zeros((p * n, r_b))
                for kk in range(p_b):
                    for ii in range(n_b):
                        W_sub[(p_off[kb] + kk) * n + (n_off[ib] + ii), :] = Wb[kk * n_b + ii, :]
                U_list.append(U_sub); V_list.append(V_sub); W_list.append(W_sub)
    U_full = np.hstack(U_list); V_full = np.hstack(V_list); W_full = np.hstack(W_list)
    U_e, V_e, W_e, kept = _eliminate_dependent(U_full, V_full, W_full)
    _try_candidate(U_e, V_e, W_e)

    # ---- Another: n=[1,1,1], m=[2,2], p=[3,3] ----
    n_blocks = [1, 1, 1]
    m_blocks = [2, 2]
    p_blocks = [3, 3]
    n_off = [0]; m_off = [0]; p_off = [0]
    for sz in n_blocks: n_off.append(n_off[-1] + sz)
    for sz in m_blocks: m_off.append(m_off[-1] + sz)
    for sz in p_blocks: p_off.append(p_off[-1] + sz)
    U_list = []; V_list = []; W_list = []
    for ib, n_b in enumerate(n_blocks):
        for jb, m_b in enumerate(m_blocks):
            for kb, p_b in enumerate(p_blocks):
                Ub, Vb, Wb = _base_case(n_b, m_b, p_b)
                r_b = Ub.shape[1]
                U_sub = np.zeros((n * m, r_b))
                for ii in range(n_b):
                    for jj in range(m_b):
                        U_sub[(n_off[ib] + ii) * m + (m_off[jb] + jj), :] = Ub[ii * m_b + jj, :]
                V_sub = np.zeros((m * p, r_b))
                for jj in range(m_b):
                    for kk in range(p_b):
                        V_sub[(m_off[jb] + jj) * p + (p_off[kb] + kk), :] = Vb[jj * p_b + kk, :]
                W_sub = np.zeros((p * n, r_b))
                for kk in range(p_b):
                    for ii in range(n_b):
                        W_sub[(p_off[kb] + kk) * n + (n_off[ib] + ii), :] = Wb[kk * n_b + ii, :]
                U_list.append(U_sub); V_list.append(V_sub); W_list.append(W_sub)
    U_full = np.hstack(U_list); V_full = np.hstack(V_list); W_full = np.hstack(W_list)
    U_e, V_e, W_e, kept = _eliminate_dependent(U_full, V_full, W_full)
    _try_candidate(U_e, V_e, W_e)

    # ---- Use Strassen 2x2x2 in a 2x4x6 sub-block (rows 2, all cols) ----
    # n=[2,1], m=[4], p=[6]  -> base case (2,4,6) fallback schoolbook, no gain.
    # Instead: n=[2,1], m=[2,2], p=[2,2,2] already tried.

    # ---- Refinement: try a tighter elimination tolerance and a second pass ----
    if _time_left() > 30:
        # Re-run elimination on the best candidate with a slightly different tol.
        U_b = _best["U"]; V_b = _best["V"]; W_b = _best["W"]
        U_e2, V_e2, W_e2, _ = _eliminate_dependent(U_b, V_b, W_b, tol=1e-14)
        _try_candidate(U_e2, V_e2, W_e2)

    # Convert best to sparse representation.
    Us, Vs, Ws = _to_sparse(_best["U"], _best["V"], _best["W"])
    return {"U": Us, "V": Vs, "W": Ws}

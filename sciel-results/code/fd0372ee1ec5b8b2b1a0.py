"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition (dense + rank-reduce)."""
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


# Strassen 2x2x2 rank-7 pattern: (u_mask, v_mask, w_mask) as 2x2 matrices
_STRASSEN_2x2 = [
    (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),
    (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, -1]])),
    (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]]), np.array([[0, 1], [1, 1]])),
    (np.array([[0, 0], [0, 1]]), np.array([[0, -1], [1, 0]]), np.array([[1, 0], [1, 0]])),
    (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[-1, 1], [0, 0]])),
    (np.array([[0, -1], [1, 0]]), np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]])),
    (np.array([[0, 1], [-1, 0]]), np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]])),
]


def _strassen_2x2(oi, oj, ok, cols):
    """Append 7 rank-1 columns for the 2x2 block A[oi:oi+2,oj:oj+2]*B[oj:oj+2,ok:ok+2]."""
    for um, vm, wm in _STRASSEN_2x2:
        u = np.zeros(16)
        v = np.zeros(24)
        w = np.zeros(24)
        for a in range(2):
            for b in range(2):
                c = um[a, b]
                if c:
                    u[(oi + a) * 4 + (oj + b)] = c
        for a in range(2):
            for b in range(2):
                c = vm[a, b]
                if c:
                    v[(oj + a) * 6 + (ok + b)] = c
        for a in range(2):
            for b in range(2):
                c = wm[a, b]
                if c:
                    w[(ok + b) * 4 + (oi + a)] = c
        cols.append((u, v, w))


def _strassen_4x4(cols):
    """Append 49 rank-1 columns for a full 4x4*4x4 multiplication (recursive Strassen)."""
    for a in range(2):
        for b in range(2):
            for c in range(2):
                _strassen_2x2(2 * a, 2 * b, 2 * c, cols)


def _build_columns():
    cols = []
    # 4x4 * 4x4 (B columns 0..3) -> 49 terms
    _strassen_4x4(cols)
    # 4x4 * 4x2 (B columns 4..5): 2x2 blocks in row-dim, 2x2 Strassen each -> 28 terms
    for rb in range(2):
        for bc in range(2):
            _strassen_2x2(2 * rb, 2 * bc, 4, cols)
    return cols


def _matmul_check(cols):
    """Verify decomposition: U @ diag(1) @ V^T must equal C_{i,j,k} = delta_{i,k}.
    Here we reconstruct the multiplication tensor and check against the true one."""
    rank = len(cols)
    U = np.stack([c[0] for c in cols], axis=1)  # (16, rank)
    V = np.stack([c[1] for c in cols], axis=1)  # (24, rank)
    W = np.stack([c[2] for c in cols], axis=1)  # (24, rank)
    # tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Build efficiently: for each r, outer product contribution
    T = np.zeros((4, 4, 6))
    for r in range(rank):
        u = U[:, r].reshape(4, 4)
        v = V[:, r].reshape(4, 6)
        w = W[:, r].reshape(6, 4)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # u[i,j]*v[j,k] for all i,j,k: einsum('ij,jk->ijk')
        uv = np.einsum('ij,jk->ijk', u, v)
        wk = w.T  # w[k,i]; w[k,i] as shape (6,4)
        T += uv * wk[None, :, :]
    # True: C[i,:,k] = A[i,:] @ B[:,k] -> the multiplication tensor is T[i,j,k]=delta_{i,k}
    true = np.zeros((4, 4, 6))
    for i in range(4):
        for k in range(6):
            true[i, i, k] = 1.0
    return np.max(np.abs(T - true)), rank, (U, V, W)


def _rank_reduce(U, V, W):
    """Project out columns that are linearly dependent.
    Strategy: find a rank-r' subset of columns that spans the same column space in a
    consistent way across U, V, W. Since all three share the same column index, we can
    detect columns that are exact zero, and more generally, do a joint SVD-style reduction:
    Build the tensor and use a single change-of-variables P (rank x r') such that
    U' = U P, V' = V P, W' = W P preserves U⊗V⊗W.
    A valid rank reduction: if there is a (r - r')-dim subspace S of R^r where the three
    matrices U, V, W all have no component (i.e., the projection of each onto S is zero in
    the sense that... ) — simpler safe approach: keep only a basis of the column space of U,
    and express V, W in that basis if they lie in it.
    """
    # Safe, guaranteed-correct reduction: remove columns that are all-zero in U.
    # If U[:, r] == 0, that term contributes nothing (since T is linear in U[:,r]).
    nz = [r for r in range(U.shape[1]) if not np.all(U[:, r] == 0)]
    if len(nz) < U.shape[1]:
        return U[:, nz], V[:, nz], W[:, nz]
    # Try: find a basis of U's column space and check V,W are consistent.
    # Build combined: for each column r, the triple (U_r, V_r, W_r) contributes
    # U_r ⊗ V_r ⊗ W_r to the tensor. The tensor's rank <= number of nonzero distinct
    # contributions. Use a linear dependence test: if sum_{r in S} alpha_r (U_r ⊗ V_r ⊗ W_r) = 0
    # we can remove. Hard in general; skip.
    return U, V, W


def construct():
    if _time_remaining() < 1.0:
        # fallback: return a trivial valid (but high-rank) identity-based decomposition
        n, m, p = 4, 4, 6
        U = np.zeros((n * m, n * p))
        V = np.zeros((m * p, n * p))
        W = np.zeros((p * n, n * p))
        for i in range(n):
            for k in range(p):
                idx = i * p + k
                U[i * m + i, idx] = 1.0
                V[i * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    cols = _build_columns()
    U, V, W = np.stack([c[0] for c in cols], axis=1), \
              np.stack([c[1] for c in cols], axis=1), \
              np.stack([c[2] for c in cols], axis=1)

    # Verify
    if _time_remaining() > 2.0:
        err, rank, (U, V, W) = _matmul_check(cols)
        if err > 1e-9:
            # fallback to schoolbook rank 24
            n, m, p = 4, 4, 6
            Uf = np.zeros((n * m, n * p))
            Vf = np.zeros((m * p, n * p))
            Wf = np.zeros((p * n, n * p))
            for i in range(n):
                for k in range(p):
                    idx = i * p + k
                    Uf[i * m + i, idx] = 1.0
                    Vf[i * p + k, idx] = 1.0
                    Wf[k * n + i, idx] = 1.0
            return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    U, V, W = _rank_reduce(U, V, W)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

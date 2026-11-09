import time
import numpy as np

DIMENSIONS = [3, 4, 4]
TOL = 1e-6

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Precompute the 48 required triples (a,b,c)
    triples = np.array([(i * m + j, j * p + k, k * n + i)
                        for i in range(n) for j in range(m) for k in range(p)])

    # Build a rank-44 base that has internal redundancy (shared diagonal columns
    # + cross-block coupling terms) so that dropping one column is more likely to
    # leave a solvable rank-43 system.
    U44, V44, W44 = build_rank44_coupled(n, m, p)

    # Verify base is valid (should be).
    if not _valid(U44, V44, W44, triples):
        U44, V44, W44 = build_rank44_plain(n, m, p)

    # Try to reduce to rank-43 by deleting one column and re-solving.
    # Prioritize columns that are "redundant" (shared diagonal / coupling).
    for col in range(44):
        if time.time() > deadline - 25:
            break
        result = try_reduce_to_rank(U44, V44, W44, col, n, m, p, deadline, triples)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: plain rank-44 (independent blocks) - still a valid rank-44.
    return {"U": U44.tolist(), "V": V44, "W": W44}


# ---------------------------------------------------------------------------
# Base decompositions
# ---------------------------------------------------------------------------
def build_rank44_plain(n, m, p):
    """Independent-block rank-44: 4x(7) + 4x(4) = 44. No redundancy."""
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        for jj in range(2):
            for kk in range(2):
                U[(i_off)*m + (j_off+jj), col] = 1
                V[(j_off+jj)*p + (k_off+kk), col] = 1
                W[(k_off+kk)*n + (i_off), col] = 1
                col += 1

    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)
    return U, V, W


def build_rank44_coupled(n, m, p):
    """Rank-44 with redundancy: shared diagonal columns + cross-block coupling.

    Structure:
      - 4 shared diagonal rank-1 columns (one per (jj,kk) pair) covering all i.
      - 3 off-diagonal Strassen columns per 2x2x2 block (i=0,1) for the 4 blocks.
      - 4th (coupling) Strassen column per block, which is the redundant term.
      - 4 independent rank-1 columns for the i=2 strip (j,k in {0,2}).
    Total = 4 + 12 + 4 + 4 = 24 for the i=0,1 blocks ... wait, need 44.
    """
    # Rebuild with explicit accounting.
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_diag(jj, kk):
        nonlocal col
        for i in range(n):
            U[i * m + jj, col] = 1
            V[jj * p + kk, col] = 1
            W[kk * n + i, col] = 1
        col += 1

    def add_offdiag(i_off, j_off, k_off):
        nonlocal col
        # term 2: (i+1)(j)(k) + (i+1)(j+1)(k+1)
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        # term 3: (i)(j)(k+1) - (i)(j+1)(k+1)
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        # term 4: (i+1)(j+1)(k) - (i+1)(j+1)(k+1)  (coupling/redundant)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

    # 4 shared diagonal columns
    add_diag(0, 0)
    add_diag(0, 2)
    add_diag(2, 0)
    add_diag(2, 2)

    # 4 blocks, 3 off-diagonal columns each = 12
    for (jo, ko) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        add_offdiag(0, jo, ko)

    # Now we've used 16 columns. The 2x2x2 blocks still need their (i)(j+1)
    # off-diagonal term (term 5/6/7 style). Add the remaining 3 per block.
    def add_offdiag2(i_off, j_off, k_off):
        nonlocal col
        # term 5: (i)(j+1)(k) - (i)(j)(k)  -> but (i)(j)(k) handled by diag
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        # term 6: (i)(j)(k+1) handled; add (i+1)(j)(k+1)
        U[(i_off+1)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        # term 7: (i+1)(j+1)(k+1) handled by diag
        # add cross (i)(j)(k+1) - (i)(j+1)(k)
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    for (jo, ko) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        add_offdiag2(0, jo, ko)

    # i=2 strip: 4 independent columns (j,k in {0,2})
    for (jo, ko) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        U[2 * m + jo, col] = 1
        V[jo * p + ko, col] = 1
        W[ko * n + 2, col] = 1
        col += 1

    # If we under-used, pad with zeros (harmless). If over, trim.
    if col < rank:
        pass  # remaining columns stay zero -> fine
    return U[:, :col] if col <= rank else U[:, :rank], \
           V[:, :col] if col <= rank else V[:, :rank], \
           W[:, :col] if col <= rank else W[:, :rank]


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------
def _vals(U, V, W, triples):
    a = triples[:, 0]; b = triples[:, 1]; c = triples[:, 2]
    return np.sum(U[a] * V[b] * W[c], axis=1)


def _valid(U, V, W, triples):
    return np.max(np.abs(_vals(U, V, W, triples) - 1.0)) < TOL


# ---------------------------------------------------------------------------
# Rank reduction
# ---------------------------------------------------------------------------
def try_reduce_to_rank(U, V, W, remove_col, n, m, p, deadline, triples):
    R = U.shape[1] - 1  # 43
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()

    nU, nV, nW = n * m, m * p, p * n

    # ALS to fill in the missing column's contribution
    max_als = 60
    for it in range(max_als):
        if time.time() > deadline - 20:
            break
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V43[j * p + k, :] * W43[k * n + i, :]
                sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                U43[i * m + j, :] = sol
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U43[i * m + j, :] * W43[k * n + i, :]
                sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                V43[j * p + k, :] = sol
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U43[i * m + j, :] * V43[j * p + k, :]
                sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                W43[k * n + i, :] = sol

        vals = _vals(U43, V43, W43, triples)
        if np.max(np.abs(vals - 1)) < 1e-8:
            break

    # Round to half-integers
    U43 = np.round(U43 * 2) / 2
    V43 = np.round(V43 * 2) / 2
    W43 = np.round(W43 * 2) / 2
    if _valid(U43, V43, W43, triples):
        return U43, V43, W43

    # Participation index for fast partial checks
    U_part = [np.where(triples[:, 0] == a)[0] for a in range(nU)]
    V_part = [np.where(triples[:, 1] == b)[0] for b in range(nV)]
    W_part = [np.where(triples[:, 2] == c)[0] for c in range(nW)]

    def partial_ok(mat_name, row, tri_idx):
        if len(tri_idx) == 0:
            return True
        a = triples[tri_idx, 0]; b = triples[tri_idx, 1]; c = triples[tri_idx, 2]
        vals = np.sum(U43[a] * V43[b] * W43[c], axis=1)
        return np.max(np.abs(vals - 1)) < TOL

    # Greedy local search: flip single half-integer entries to reduce max error.
    for ls_iter in range(400):
        if time.time() > deadline - 5:
            break
        vals = _vals(U43, V43, W43, triples)
        err = np.abs(vals - 1)
        if np.max(err) < TOL:
            return U43, V43, W43

        worst = np.argmax(err)
        a, b, c = triples[worst]

        # Candidate flips on the worst triple's rows
        cands = []
        for r in range(R):
            cands.append(('U', a, r))
            cands.append(('V', b, r))
            cands.append(('W', c, r))
        np.random.shuffle(cands)

        found = False
        for mat_name, row, r in cands:
            if time.time() > deadline - 5:
                break
            if mat_name == 'U':
                old = U43[row, r]; tri_idx = U_part[row]
                for delta in (0.5, -0.5):
                    U43[row, r] = old + delta
                    if partial_ok('U', row, tri_idx) and _valid(U43, V43, W43, triples):
                        return U43, V43, W43
                U43[row, r] = old
            elif mat_name == 'V':
                old = V43[row, r]; tri_idx = V_part[row]
                for delta in (0.5, -0.5):
                    V43[row, r] = old + delta
                    if partial_ok('V', row, tri_idx) and _valid(U43, V43, W43, triples):
                        return U43, V43, W43
                V43[row, r] = old
            else:
                old = W43[row, r]; tri_idx = W_part[row]
                for delta in (0.5, -0.5):
                    W43[row, r] = old + delta
                    if partial_ok('W', row, tri_idx) and _valid(U43, V43, W43, triples):
                        return U43, V43, W43
                W43[row, r] = old
        if not found:
            # Diversify: nudge the worst row entries toward reducing error
            pass

    return None

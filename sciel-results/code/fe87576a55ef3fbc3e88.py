import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]


def construct():
    n, m, p = 3, 4, 5
    start = time.time()
    deadline = start + 135.0

    # ---------- verification (trace convention W[k*n+i, r]) ----------
    def verify(U, V, W, tol=1e-7):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        R = U.shape[1]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, m))
                for r in range(R):
                    w = W3[k, i, r]
                    if w != 0:
                        M += w * np.outer(U3[i, :, r], V3[:, k, r])
                if not np.allclose(M, np.eye(m), atol=tol):
                    return False
        return True

    # ---------- Strassen 2x2x2 block ----------
    def strassen_block(i0, j0, k0, a, b, c):
        # returns list of (i, j, k, u, v, w)
        terms = [
            (i0 + 0, j0 + 0, k0 + 0, a, b, c),
            (i0 + 0, j0 + 1, k0 + 0, a, b, -c),
            (i0 + 1, j0 + 0, k0 + 0, -a, -b, c),
            (i0 + 1, j0 + 1, k0 + 0, a, b, c),
            (i0 + 0, j0 + 0, k0 + 1, a, -b, c),
            (i0 + 0, j0 + 1, k0 + 1, a, b, c),
            (i0 + 1, j0 + 0, k0 + 1, -a, -b, -c),
            (i0 + 1, j0 + 1, k0 + 1, a, b, -c),
        ]
        return terms

    # ---------- schoolbook term ----------
    def schoolbook_term(i, j, k):
        return (i, j, k, 1, 1, 1)

    # ---------- build full term list ----------
    def build_terms(blocks, school_idx):
        terms = []
        for blk in blocks:
            terms.extend(blk)
        for (i, j, k) in school_idx:
            terms.append(schoolbook_term(i, j, k))
        return terms

    def term_index(i, j, k):
        return (i * m + j) * (m * p) + (j * p + k)

    # ---------- matrix representation ----------
    def term_matrix(terms):
        # rows: (i,k) in [n*p]; cols: (j,a) in [m*m]; size R
        rows = n * p
        cols = m * m
        T = np.zeros((rows, cols, len(terms)))
        for r, (i, j, k, u, v, w) in enumerate(terms):
            T[k * n + i, j * m + (j), r] = u * v * w
        return T

    def rank_of(T):
        # T: (rows, cols, R)
        rows, cols, R = T.shape
        M = T.reshape(rows * cols, R)
        try:
            return np.linalg.matrix_rank(M, tol=1e-8)
        except np.linalg.LinAlgError:
            return R

    # ---------- dependency search (drop one term via absorption) ----------
    def find_dependency(T):
        rows, cols, R = T.shape
        M = T.reshape(rows * cols, R)
        # find a null vector
        try:
            _, s, Vh = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            return None
        # smallest singular value
        idx = int(np.argmin(s))
        if s[idx] > 1e-7:
            return None
        c = Vh[idx]
        # find a component with large magnitude to drop
        am = np.abs(c)
        r_drop = int(np.argmax(am))
        if abs(c[r_drop]) < 1e-7:
            return None
        return c, r_drop

    # ---------- assemble factor matrices from terms ----------
    def assemble(terms):
        R = len(terms)
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        for r, (i, j, k, u, v, w) in enumerate(terms):
            U[i * m + j, r] = u
            V[j * p + k, r] = v
            W[k * n + i, r] = w
        return U, V, W

    # ---------- candidate blocks (all 2x2x2 within 3x4x5) ----------
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))

    # ---------- greedy disjoint block selection ----------
    def block_cells(i0, j0, k0):
        cells = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    cells.add((i0 + di, j0 + dj, k0 + dk))
        return cells

    def greedy_blocks():
        # try all orders via simple greedy by index; also try a few fixed orders
        best = None
        for order_key in ['default', 'rev']:
            order = list(range(len(all_blocks)))
            if order_key == 'rev':
                order = order[::-1]
            used = set()
            chosen = []
            for bi in order:
                cells = block_cells(*all_blocks[bi])
                if not (cells & used):
                    used |= cells
                    chosen.append(all_blocks[bi])
            if best is None or len(chosen) > len(best):
                best = chosen
        return best

    def schoolbook_remainder(chosen):
        used = set()
        for (i0, j0, k0) in chosen:
            used |= block_cells(i0, j0, k0)
        rem = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in used:
                        rem.append((i, j, k))
        return rem

    # ---------- main: build base hybrid (4 blocks) ----------
    chosen = greedy_blocks()
    # ensure we use up to 4 disjoint blocks
    if len(chosen) < 4:
        # pad with more if possible (shouldn't happen)
        pass
    rem = schoolbook_remainder(chosen)
    base_blocks = chosen[:4] if len(chosen) >= 4 else chosen

    # base terms (a=b=c=1)
    base_terms = []
    for blk in base_blocks:
        base_terms.extend(strassen_block(blk[0], blk[1], blk[2], 1, 1, 1))
    for (i, j, k) in rem:
        base_terms.append(schoolbook_term(i, j, k))

    base_U, base_V, base_W = assemble(base_terms)
    base_rank = len(base_terms)
    best = (base_rank, base_U, base_V, base_W)

    # ---------- search over Strassen coefficient variants for a dependency ----------
    # Each block has 7 multipliers (a,b,c per block). Search 7^7 grid is too big;
    # instead try: fix blocks 0..2 at (1,1,1), vary block 3, then vary block 2, etc.
    # We'll iterate over a set of coefficient triples per block.
    coef_choices = [(-2, -1, 1, 2)]  # per-multiplier values

    # Build a search space: for each block, choose (a,b,c) from a small set.
    small = [(-2, -1, 1, 2)]
    abc_choices = list(itertools.product(small, small, small))  # 64 per block

    # Strategy: vary one block at a time (keep others at (1,1,1)), then two blocks.
    nblocks = len(base_blocks)

    def make_terms_from_coefs(coefs):
        terms = []
        for bi, blk in enumerate(base_blocks):
            a, b, c = coefs[bi]
            terms.extend(strassen_block(blk[0], blk[1], blk[2], a, b, c))
        for (i, j, k) in rem:
            terms.append(schoolbook_term(i, j, k))
        return terms

    default_coefs = [(1, 1, 1)] * nblocks

    # Precompute the fixed part (schoolbook + blocks held at default) as a matrix
    # For speed, we recompute full T each iteration but it's small (R~56, 60x16).

    # Iteration 1: vary each single block over all 64 abc choices
    for bi in range(nblocks):
        if time.time() > deadline:
            break
        for abc in abc_choices:
            if time.time() > deadline:
                break
            coefs = list(default_coefs)
            coefs[bi] = abc
            terms = make_terms_from_coefs(coefs)
            T = term_matrix(terms)
            dep = find_dependency(T)
            if dep is not None:
                c, r_drop = dep
                # absorb: for r != r_drop, terms[r] *= c[r]/c[r_drop]; drop r_drop
                new_terms = []
                scale = c[r_drop]
                for r in range(len(terms)):
                    if r == r_drop:
                        continue
                    i, j, k, u, v, w = terms[r]
                    f = c[r] / scale
                    new_terms.append((i, j, k, u * f, v * f, w * f))
                new_terms.append((0, 0, 0, 1, 1, 1))  # placeholder dropped term (zeroed below)
                # actually the dropped term is removed; the absorbed term is r_drop with coefficient 1
                # rebuild properly: keep r_drop term with its own (u,v,w) scaled by 1, others scaled
                new_terms = []
                for r in range(len(terms)):
                    i, j, k, u, v, w = terms[r]
                    if r == r_drop:
                        new_terms.append((i, j, k, u, v, w))
                    else:
                        f = c[r] / scale
                        new_terms.append((i, j, k, u * f, v * f, w * f))
                U2, V2, W2 = assemble(new_terms)
                if verify(U2, V2, W2, tol=1e-6):
                    nr = len(new_terms)
                    if nr < best[0]:
                        best = (nr, U2, V2, W2)

    # Iteration 2: vary two blocks (coarser: subset of choices)
    if time.time() < deadline - 20:
        coarse = [(-1, 1), (1, -1), (-2, 1), (1, 2), (2, -1)]
        coarse_abc = []
        for a in (-2, -1, 1, 2):
            for b in (-2, -1, 1, 2):
                for c in (-2, -1, 1, 2):
                    coarse_abc.append((a, b, c))
        # limit to a subset to keep time bounded
        coarse_abc = coarse_abc[::3]  # ~21 choices
        for bi in range(nblocks):
            for bj in range(bi + 1, nblocks):
                if time.time() > deadline:
                    break
                for abc_i in coarse_abc:
                    for abc_j in coarse_abc:
                        if time.time() > deadline:
                            break
                        coefs = list(default_coefs)
                        coefs[bi] = abc_i
                        coefs[bj] = abc_j
                        terms = make_terms_from_coefs(coefs)
                        T = term_matrix(terms)
                        dep = find_dependency(T)
                        if dep is not None:
                            c, r_drop = dep
                            scale = c[r_drop]
                            new_terms = []
                            for r in range(len(terms)):
                                i, j, k, u, v, w = terms[r]
                                if r == r_drop:
                                    new_terms.append((i, j, k, u, v, w))
                                else:
                                    f = c[r] / scale
                                    new_terms.append((i, j, k, u * f, v * f, w * f))
                            U2, V2, W2 = assemble(new_terms)
                            if verify(U2, V2, W2, tol=1e-6):
                                nr = len(new_terms)
                                if nr < best[0]:
                                    best = (nr, U2, V2, W2)

    # ---------- final: round to half-integers and re-verify ----------
    rank, U, V, W = best
    # try to snap to half-integers if close
    def snap(M):
        M2 = 2 * M
        if np.allclose(M2, np.round(M2), atol=1e-5):
            return np.round(M2) / 2.0
        return M

    U = snap(U)
    V = snap(V)
    W = snap(W)
    if not verify(U, V, W, tol=1e-7):
        # fall back to base (unsnapped)
        U, V, W = base_U, base_V, base_W
        rank = base_rank

    # ---------- fallback: pure schoolbook if something went wrong ----------
    if not verify(U, V, W, tol=1e-7):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = R

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

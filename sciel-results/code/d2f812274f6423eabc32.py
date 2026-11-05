"""Tensor rank minimization for 4x5 x 5x6 matrix multiplication via block
decomposition with Strassen on 2x2x2 blocks."""
import time

DIMENSIONS = [4, 5, 6]
_DEADLINE = time.time() + 140.0


def _expired():
    return time.time() > _DEADLINE


def _gen_strassen222():
    """Yield (u, v, w) for 2x2x2 Strassen.
    u: len 4 (A entries, index r*2+c), v: len 4 (B entries, index r*2+c),
    w: len 4 (C entries, index r*2+c)."""
    def A(s):
        v = [0]*4
        for (r, c, sgn) in s:
            v[r*2+c] += sgn
        return v
    def B(s):
        v = [0]*4
        for (r, c, sgn) in s:
            v[r*2+c] += sgn
        return v
    def C(s):
        v = [0]*4
        for (r, c, sgn) in s:
            v[r*2+c] += sgn
        return v
    terms = [
        # M1 = (a11+a22)(b11+b22) -> c11(+1), c22(+1)
        (A([(0,0,1),(1,1,1)]), B([(0,0,1),(1,1,1)]), C([(0,0,1),(1,1,1)])),
        # M2 = (a21+a22)b11 -> c21(+1), c22(-1)
        (A([(1,0,1),(1,1,1)]), B([(0,0,1)]), C([(1,0,1),(1,1,-1)])),
        # M3 = a11(b12-b22) -> c12(+1), c22(+1)
        (A([(0,0,1)]), B([(0,1,1),(1,1,-1)]), C([(0,1,1),(1,1,1)])),
        # M4 = a22(b21-b11) -> c11(+1), c21(+1)
        (A([(1,1,1)]), B([(1,0,1),(0,0,-1)]), C([(0,0,1),(1,0,1)])),
        # M5 = (a11+a12)b22 -> c11(-1), c12(+1)
        (A([(0,0,1),(0,1,1)]), B([(1,1,1)]), C([(0,0,-1),(0,1,1)])),
        # M6 = (a21-a11)(b11+b12) -> c22(+1)
        (A([(1,0,1),(0,0,-1)]), B([(0,0,1),(0,1,1)]), C([(1,1,1)])),
        # M7 = (a12-a22)(b21+b22) -> c11(+1), c22(+1)
        (A([(0,1,1),(1,1,-1)]), B([(1,0,1),(1,1,1)]), C([(0,0,1),(1,1,1)])),
    ]
    for t in terms:
        yield t


def _gen_trivial(nb, mb, pb):
    nA, nB, nC = nb*mb, mb*pb, nb*pb
    for a in range(nb):
        for k in range(mb):
            for d in range(pb):
                u = [0]*nA; u[a*mb+k] = 1
                v = [0]*nB; v[k*pb+d] = 1
                w = [0]*nC; w[a*pb+d] = 1
                yield (u, v, w)


def _block_gen(nb, mb, pb):
    if nb == 2 and mb == 2 and pb == 2:
        return _gen_strassen222()
    return _gen_trivial(nb, mb, pb)


def _splits(dim):
    out = []
    for a in range(1, dim):
        out.append((a, dim-a))
    for a in range(1, dim-1):
        for b in range(1, dim-a):
            out.append((a, b, dim-a-b))
    return out


def _build(n, m, p, sn, sm, sp):
    U_rows, V_rows, W_rows = n*m, m*p, p*n
    U, V, W = [], [], []
    n_off = 0
    for nb in sn:
        m_off = 0
        for mb in sm:
            p_off = 0
            for pb in sp:
                for u, v, w in _block_gen(nb, mb, pb):
                    uc = [0]*U_rows
                    for idx, val in enumerate(u):
                        if val:
                            a, k = divmod(idx, mb)
                            uc[(n_off+a)*m + (m_off+k)] = val
                    vc = [0]*V_rows
                    for idx, val in enumerate(v):
                        if val:
                            k, d = divmod(idx, pb)
                            vc[(m_off+k)*p + (p_off+d)] = val
                    wc = [0]*W_rows
                    for idx, val in enumerate(w):
                        if val:
                            a, d = divmod(idx, pb)
                            wc[(p_off+d)*n + (n_off+a)] = val
                    U.append(uc); V.append(vc); W.append(wc)
                p_off += pb
            m_off += mb
        n_off += nb
    return U, V, W, len(U)


def construct():
    n, m, p = DIMENSIONS
    best, best_rank = None, n*m*p
    for sn in _splits(n):
        if _expired(): break
        for sm in _splits(m):
            if _expired(): break
            for sp in _splits(p):
                if _expired(): break
                U, V, W, r = _build(n, m, p, sn, sm, sp)
                if r < best_rank:
                    best_rank, best = r, (U, V, W)
    if best is None:
        r = n*m*p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1; V[j*p+k][c] = 1; W[k*n+i][c] = 1; c += 1
        return {"U": U, "V": V, "W": W}
    U, V, W = best
    return {"U": U, "V": V, "W": W}

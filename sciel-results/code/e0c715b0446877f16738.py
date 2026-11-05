"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the (n*m) x (m*p) x (p*n) tensor T where T[a,b,c]=1 iff consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return bool(np.allclose(result, T, atol=tol))

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition (always valid)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = n*m*p
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _rank17(n, m, p):
    """Verified Strassen(2x2x2) + direct decomposition of rank 17."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = 17
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))

    def set_term(r, u, v, w):
        for a, val in u.items(): U[a, r] = val
        for b, val in v.items(): V[b, r] = val
        for c, val in w.items(): W[c, r] = val

    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2,0:2]
    set_term(0, {0:1,4:1}, {0:1,4:1}, {0:1,3:1})   # M1=(a0+a4)(b0+b4)
    set_term(1, {3:1,4:1}, {0:1},       {1:1,3:-1}) # M2=(a3+a4)b0
    set_term(2, {0:1},       {1:1,4:-1}, {2:1,3:1}) # M3=a0(b1-b4)
    set_term(3, {4:1},       {3:1,0:-1}, {0:1,1:1}) # M4=a4(b3-b0)
    set_term(4, {0:1,1:1},   {4:1},      {0:-1,2:1})# M5=(a0+a1)b4
    set_term(5, {3:1,0:-1},  {0:1,1:1},  {3:1})     # M6=(a3-a0)(b0+b1)
    set_term(6, {1:1,4:-1},  {3:1,4:1},  {0:1})     # M7=(a1-a4)(b3+b4)
    # Part 2: A1(2x2) x B1_right(2x1) -> C[:,2]
    set_term(7,  {0:1}, {2:1}, {4:1})
    set_term(8,  {1:1}, {5:1}, {4:1})
    set_term(9,  {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    # Part 3: A2(2x1) x B2(1x3) -> C[0:2,0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    return U, V, W

def _try_reduced(T, U, V, W, R, deadline):
    """
    Bounded search for a strictly smaller rank decomposition.
    Only returns a solution if it is EXACTLY verified; otherwise returns None.
    """
    n_a, n_b, n_c = T.shape
    rank = U.shape[1]
    best = None

    def matmul_of_terms(keep, extra):
        cols = list(keep)
        Ux = np.zeros((n_a, len(cols) + len(extra)))
        Vx = np.zeros((n_b, len(cols) + len(extra)))
        Wx = np.zeros((n_c, len(cols) + len(extra)))
        for ci, ti in enumerate(cols):
            Ux[:, ci] = U[:, ti]; Vx[:, ci] = V[:, ti]; Wx[:, ci] = W[:, ti]
        for ei, (uu, vv, ww) in enumerate(extra):
            Ux[:, len(cols)+ei] = uu; Vx[:, len(cols)+ei] = vv; Wx[:, len(cols)+ei] = ww
        return Ux, Vx, Wx

    grid = [-2, -1, 0, 1, 2]

    # (a) single-term drop: try removing each term
    for drop in range(rank):
        if time.time() > deadline: break
        keep = [t for t in range(rank) if t != drop]
        Ux, Vx, Wx = matmul_of_terms(keep, [])
        if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
            best = (Ux, Vx, Wx)
            break

    if best is not None:
        return best

    # (b) drop two terms, add one new rank-1 term from a small structured grid.
    for d1 in range(rank):
        if time.time() > deadline: break
        for d2 in range(d1+1, rank):
            if time.time() > deadline: break
            keep = [t for t in range(rank) if t not in (d1, d2)]
            for ca in grid:
                for cb in grid:
                    if ca == 0 and cb == 0:
                        continue
                    uu = ca*U[:, d1] + cb*U[:, d2]
                    vv = ca*V[:, d1] + cb*V[:, d2]
                    ww = ca*W[:, d1] + cb*W[:, d2]
                    Ux, Vx, Wx = matmul_of_terms(keep, [(uu, vv, ww)])
                    if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
                        return (Ux, Vx, Wx)

    # (c) Drop 1 term, try to replace with a new term that is a linear combination
    #     of multiple existing terms (not just the dropped one).
    #     For each term to drop, try replacing it with a combo of 2-3 other terms.
    for drop in range(rank):
        if time.time() > deadline: break
        keep = [t for t in range(rank) if t != drop]
        # Try adding back a term that's a combo of 2 kept terms
        for i in range(len(keep)):
            if time.time() > deadline: break
            for j in range(i+1, len(keep)):
                if time.time() > deadline: break
                t1, t2 = keep[i], keep[j]
                for ca in [-1, 1]:
                    for cb in [-1, 1]:
                        uu = ca*U[:, t1] + cb*U[:, t2]
                        vv = ca*V[:, t1] + cb*V[:, t2]
                        ww = ca*W[:, t1] + cb*W[:, t2]
                        Ux, Vx, Wx = matmul_of_terms(keep, [(uu, vv, ww)])
                        if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
                            return (Ux, Vx, Wx)

    # (d) Drop 2 terms, try replacing with a combo of 2 other terms
    for d1 in range(rank):
        if time.time() > deadline: break
        for d2 in range(d1+1, rank):
            if time.time() > deadline: break
            keep = [t for t in range(rank) if t not in (d1, d2)]
            for i in range(len(keep)):
                if time.time() > deadline: break
                for j in range(i+1, len(keep)):
                    if time.time() > deadline: break
                    t1, t2 = keep[i], keep[j]
                    for ca in [-1, 1]:
                        for cb in [-1, 1]:
                            uu = ca*U[:, t1] + cb*U[:, t2]
                            vv = ca*V[:, t1] + cb*V[:, t2]
                            ww = ca*W[:, t1] + cb*W[:, t2]
                            Ux, Vx, Wx = matmul_of_terms(keep, [(uu, vv, ww)])
                            if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
                                return (Ux, Vx, Wx)

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # explicit safety margin (< 120s wall-clock)

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Baseline: verified rank-17 (Strassen + direct).
    U17, V17, W17 = _rank17(n, m, p)
    if _verify_decomposition(U17, V17, W17, T):
        base = (U17, V17, W17)
    else:
        base = _schoolbook(n, m, p)

    # Bounded refinement: attempt a strictly smaller, EXACTLY verified rank.
    # This never hurts: it only replaces the baseline if verified.
    reduced = _try_reduced(T, base[0], base[1], base[2], base[0].shape[1], deadline)
    if reduced is not None:
        U, V, W = reduced
    else:
        U, V, W = base

    # Final exact verification; fall back to schoolbook on any failure.
    if not _verify_decomposition(U, V, W, T):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

"""Tensor decomposition for 3x4 x 4x5 matrix multiplication (rank minimization)."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_system(n, m, p):
    """Build the full linear system A x = b over all (a,b,c) tensor entries.

    The matrix multiplication tensor T[a,b,c] = 1 iff (i,j,k) is consistent
    (a=i*m+j, b=j*p+k, c=k*n+i), else 0.
    We encode each entry as a linear equation in x = concat(U.ravel(), V.ravel(), W.ravel()).
    Returns (A, b) with A of shape (n*m*m*p*p*n, n*m*r + m*p*r + p*n*r).
    """
    nm, mp, pn = n * m, m * p, p * n
    N = nm * mp * pn
    # precompute row index for each (a,b,c)
    # For entry (a,b,c): a=i*m+j, b=j'*p+k, c=i'*n+k'
    # nonzero iff i==i', j==j', k==k'
    b = np.zeros(N)
    # Build list of (a, b, c, val)
    triples = []
    for a in range(nm):
        i = a // m
        j = a % m
        for bb in range(mp):
            jp = bb // p
            kk = bb % p
            for cc in range(pn):
                ii = cc // n
                kk2 = cc % n
                val = 1.0 if (i == ii and j == jp and kk == kk2) else 0.0
                row = (a * mp + bb) * pn + cc
                if val != 0.0:
                    b[row] = 1.0
                triples.append((a, bb, cc, val))
    return nm, mp, pn, triples, b


def _make_A_for_W(n, m, p, nm, mp, pn, triples, r):
    """Build A (N x (nm*r + mp*r)) given W fixed.
    Equation for entry (a,b,c): sum_q U[a,q]*V[b,q]*W[c,q] = b_entry
    => sum_q (W[c,q] * U[a,q]) * V[b,q] = b_entry
    Variables: U (nm*r) then V (mp*r).
    Coefficient for U[a,q] is V[b,q]*W[c,q]; for V[b,q] is U[a,q]*W[c,q].
    This is bilinear, not linear! So we cannot fix W and make it linear in U,V jointly.
    """
    raise NotImplementedError


def _als_step_U(A_U, b_U):
    """Solve least squares min ||A_U x_U - b_U||^2."""
    # A_U: (N, nm*r)
    try:
        x, *_ = np.linalg.lstsq(A_U, b_U, rcond=None)
        return x
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(A_U) @ b_U
        return x


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n
    N = nm * mp * pn  # 3600

    # Build b (the target tensor entries)
    b = np.zeros(N)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                bb = j * p + k
                cc = k * n + i
                row = (a * mp + bb) * pn + cc
                b[row] = 1.0

    # Precompute index arrays
    # a_idx[entry] = a, b_idx[entry] = bb, c_idx[entry] = cc
    a_idx = np.zeros(N, dtype=int)
    b_idx = np.zeros(N, dtype=int)
    c_idx = np.zeros(N, dtype=int)
    for a in range(nm):
        for bb in range(mp):
            for cc in range(pn):
                row = (a * mp + bb) * pn + cc
                a_idx[row] = a
                b_idx[row] = bb
                c_idx[row] = cc

    def reconstruct(U, V, W):
        """Reconstruct all N tensor entries. U:(nm,r), V:(mp,r), W:(pn,r)."""
        # entry(row) = sum_q U[a,q]*V[bb,q]*W[cc,q]
        # = sum_q (U[a,q]*V[bb,q]) * W[cc,q]
        # Compute M1[a,bb,q] = U[a,q]*V[bb,q], then entry = M1 @ W per (a,bb)
        # Efficient: for each q, outer = U[:,q][:,None] * V[:,q][None,:]  (nm x mp)
        # then entry += outer * W[cc,q] broadcast over cc
        out = np.zeros(N)
        r = U.shape[1]
        for q in range(r):
            # M1: (nm, mp)
            M1 = U[:, q][:, None] * V[:, q][None, :]
            # For each (a,bb), entry over cc = M1[a,bb] * W[:,q]
            # entry rows for given (a,bb) are contiguous block of pn
            out += (M1[:, :, None] * W[:, q][None, None, :]).reshape(N)
        return out

    def verify(U, V, W, tol=1e-7):
        out = reconstruct(U, V, W)
        return np.max(np.abs(out - b)) <= tol

    # Schoolbook fallback
    rank_sb = n * m * p  # 60
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    np.random.seed(42)

    # ALS: fix two factors, solve linear least squares for the third.
    # When W is fixed, the equation for entry (a,bb,cc) is:
    #   sum_q U[a,q]*V[bb,q]*W[cc,q] = b
    # This is linear in the products... but not jointly linear in U and V.
    # Standard ALS for CP: fix V and W, solve for U (linear), etc.
    # For U: entry = sum_q U[a,q] * (V[bb,q]*W[cc,q]) = b
    #   => U[a,q] * (V[bb,q]*W[cc,q]) summed over q = b
    #   Linear in U! Coefficient of U[a,q] is V[bb,q]*W[cc,q].
    # So A_U[a*q + q? ] ... let's build A_U of shape (N, nm*r).
    # Row = entry (a,bb,cc); column = (a', q). Coeff nonzero only if a'==a: V[bb,q]*W[cc,q].

    def solve_U(V, W, r):
        # A_U: (N, nm*r)
        # For entry (a,bb,cc), column index for (a,q) is a*r+q, coeff = V[bb,q]*W[cc,q]
        A_U = np.zeros((N, nm * r))
        for row in range(N):
            a = a_idx[row]
            bb = b_idx[row]
            cc = c_idx[row]
            for q in range(r):
                A_U[row, a * r + q] = V[bb, q] * W[cc, q]
        x, *_ = np.linalg.lstsq(A_U, b, rcond=None)
        return x.reshape(nm, r)

    def solve_V(U, W, r):
        A_V = np.zeros((N, mp * r))
        for row in range(N):
            a = a_idx[row]
            bb = b_idx[row]
            cc = c_idx[row]
            for q in range(r):
                A_V[row, bb * r + q] = U[a, q] * W[cc, q]
        x, *_ = np.linalg.lstsq(A_V, b, rcond=None)
        return x.reshape(mp, r)

    def solve_W(U, V, r):
        A_W = np.zeros((N, pn * r))
        for row in range(N):
            a = a_idx[row]
            bb = b_idx[row]
            cc = c_idx[row]
            for q in range(r):
                A_W[row, cc * r + q] = U[a, q] * V[bb, q]
        x, *_ = np.linalg.lstsq(A_W, b, rcond=None)
        return x.reshape(pn, r)

    def run_als(r, init_U, init_V, init_W, max_iter, time_limit):
        U, V, W = init_U.copy(), init_V.copy(), init_W.copy()
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            U = solve_U(V, W, r)
            V = solve_V(U, W, r)
            W = solve_W(U, V, r)
            if verify(U, V, W, tol=1e-7):
                break
        return U, V, W

    # Try ranks from schoolbook-1 downward, but focus effort on a few promising ranks.
    # The true bilinear rank of M_{3,4,5} is unknown to us; try a range.
    candidate_ranks = list(range(rank_sb - 1, max(20, rank_sb - 20), -1))

    for r in candidate_ranks:
        if time.time() > deadline - 8:
            break
        found = False
        for attempt in range(4):
            if time.time() > deadline - 8:
                break
            scale = 0.5
            U0 = np.random.randn(nm, r) * scale
            V0 = np.random.randn(mp, r) * scale
            W0 = np.random.randn(pn, r) * scale
            # Seed a few columns with schoolbook pattern to give structure
            for q in range(min(3, r)):
                i = q % n
                j = (q // n) % m
                k = (q // (n * m)) % p
                U0[i * m + j, q] += 1.0
                V0[j * p + k, q] += 1.0
                W0[k * n + i, q] += 1.0
            U, V, W = run_als(r, U0, V0, W0, max_iter=60, time_limit=deadline - 8)
            if verify(U, V, W, tol=1e-7):
                # Polish: more iterations, tighter
                for _ in range(15):
                    U = solve_U(V, W, r)
                    V = solve_V(U, W, r)
                    W = solve_W(U, V, r)
                if verify(U, V, W, tol=1e-9):
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        if found:
            break

    # Final verification
    if best_rank < rank_sb:
        if verify(best_U, best_V, best_W, tol=1e-7):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

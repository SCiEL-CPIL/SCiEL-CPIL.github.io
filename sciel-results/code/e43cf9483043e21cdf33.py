import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _round_half(x):
    h = x * 2
    r = round(h)
    if abs(h - r) < 1e-6:
        return Fraction(r, 2)
    ri = round(x)
    if abs(x - ri) < 1e-6:
        return Fraction(ri)
    return Fraction(x)


def _try_rank_structured(n, m, p, R, seed, max_iters=400, anchor_kind=0):
    """
    Generalized structured rank-R search.
    Fix one anchor term (r=0) to break scale symmetry, then use alternating
    linear least squares for the remaining R-1 terms.

    anchor_kind:
      0: anchor covers (i=0, k=0) for all j  [the original]
      1: anchor covers (i=1, k=0) for all j
      2: anchor covers (i=0, k=1) for all j
      3: anchor covers a single (i,j,k) = (0,0,0)
    """
    rng = random.Random(seed)

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Build anchor term
    u0 = [0.0] * (n * m)
    v0 = [0.0] * (m * p)
    w0 = [0.0] * (p * n)

    if anchor_kind == 0:
        for j in range(m):
            u0[u_idx(0, j)] = 1.0
            v0[v_idx(j, 0)] = 1.0
        w0[w_idx(0, 0)] = 1.0
    elif anchor_kind == 1:
        for j in range(m):
            u0[u_idx(1, j)] = 1.0
            v0[v_idx(j, 0)] = 1.0
        w0[w_idx(0, 1)] = 1.0
    elif anchor_kind == 2:
        for j in range(m):
            u0[u_idx(0, j)] = 1.0
            v0[v_idx(j, 1)] = 1.0
        w0[w_idx(1, 0)] = 1.0
    else:  # single point
        u0[u_idx(0, 0)] = 1.0
        v0[v_idx(0, 0)] = 1.0
        w0[w_idx(0, 0)] = 1.0

    # Determine which (i,j,k) are covered by anchor (anchor contributes exactly 1 there)
    covered = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                contrib = u0[u_idx(i, j)] * v0[v_idx(j, k)] * w0[w_idx(k, i)]
                if abs(contrib) > 1e-9:
                    if abs(contrib - 1.0) < 1e-9:
                        covered.add((i, j, k))

    # All targets
    all_targets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    # Targets that the anchor must cover exactly
    # For a valid decomposition, the anchor's contribution to any (i,j,k) not in covered must be 0
    # (it is, by construction). For (i,j,k) in covered, the anchor gives 1, so remaining terms must give 0.
    # For (i,j,k) not in covered, remaining terms must give 1.

    # Initialize u, v, w
    u = [[0.0] * (n * m) for _ in range(R)]
    v = [[0.0] * (m * p) for _ in range(R)]
    w = [[0.0] * (p * n) for _ in range(R)]
    u[0] = u0[:]
    v[0] = v0[:]
    w[0] = w0[:]

    # Initialize remaining terms
    # For each target not covered, create a dedicated term (up to R-1 terms)
    uncovered = [t for t in all_targets if t not in covered]
    for idx, (i, j, k) in enumerate(uncovered[:R - 1]):
        r = idx + 1
        u[r][u_idx(i, j)] = 1.0
        v[r][v_idx(j, k)] = 1.0
        w[r][w_idx(k, i)] = 1.0

    # Any leftover terms (if R-1 > len(uncovered)) get small random init
    for r in range(1 + len(uncovered[:R - 1]), R):
        for i in range(n * m):
            u[r][i] = rng.uniform(-0.3, 0.3)
        for j in range(m * p):
            v[r][j] = rng.uniform(-0.3, 0.3)
        for k in range(p * n):
            w[r][k] = rng.uniform(-0.3, 0.3)

    def compute_residual(u, v, w):
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += u[r][u_idx(i, j)] * v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    res += (s - 1.0) ** 2
        return res ** 0.5

    def _rescale(u, v, w, factor=1.0):
        """Scale all non-anchor terms by a factor to keep magnitudes reasonable."""
        for r in range(1, R):
            for i in range(n * m):
                u[r][i] *= factor
            for j in range(m * p):
                v[r][j] *= factor
            for k in range(p * n):
                w[r][k] *= factor

    def solve_u(v, w):
        A = np.zeros((n * m * p, (R - 1) * n * m))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs[eq] = 1.0 - u[0][u_idx(i, j)] * v[0][v_idx(j, k)] * w[0][w_idx(k, i)]
                    for r in range(1, R):
                        A[eq, (r - 1) * n * m + i * m + j] = v[r][v_idx(j, k)] * w[r][w_idx(k, i)]
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        u_new = [row[:] for row in u]
        for r in range(1, R):
            for i in range(n):
                for j in range(m):
                    u_new[r][i * m + j] = sol[(r - 1) * n * m + i * m + j]
        return u_new

    def solve_v(u, w):
        A = np.zeros((n * m * p, (R - 1) * m * p))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs[eq] = 1.0 - u[0][u_idx(i, j)] * v[0][v_idx(j, k)] * w[0][w_idx(k, i)]
                    for r in range(1, R):
                        A[eq, (r - 1) * m * p + j * p + k] = u[r][u_idx(i, j)] * w[r][w_idx(k, i)]
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        v_new = [row[:] for row in v]
        for r in range(1, R):
            for j in range(m):
                for k in range(p):
                    v_new[r][j * p + k] = sol[(r - 1) * m * p + j * p + k]
        return v_new

    def solve_w(u, v):
        A = np.zeros((n * m * p, (R - 1) * p * n))
        rhs = np.zeros(n * m * p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs[eq] = 1.0 - u[0][u_idx(i, j)] * v[0][v_idx(j, k)] * w[0][w_idx(k, i)]
                    for r in range(1, R):
                        A[eq, (r - 1) * p * n + k * n + i] = u[r][u_idx(i, j)] * v[r][v_idx(j, k)]
                    eq += 1
        try:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except Exception:
            return None
        w_new = [row[:] for row in w]
        for r in range(1, R):
            for k in range(p):
                for i in range(n):
                    w_new[r][k * n + i] = sol[(r - 1) * p * n + k * n + i]
        return w_new

    best_res = float('inf')
    best_state = None
    plateau_count = 0

    for it in range(max_iters):
        if deadline_exceeded():
            break

        u_new = solve_u(v, w)
        if u_new is None:
            break
        u = u_new

        v_new = solve_v(u, w)
        if v_new is None:
            break
        v = v_new

        w_new = solve_w(u, v)
        if w_new is None:
            break
        w = w_new

        # Periodic rescaling to prevent drift
        if it % 20 == 19 and it > 0:
            _rescale(u, v, w, 0.95)

        res = compute_residual(u, v, w)

        if res < best_res - 1e-12:
            best_res = res
            best_state = (u[:], [row[:] for row in v], [row[:] for row in w])
            plateau_count = 0
            if res < 1e-10:
                break
        else:
            plateau_count += 1
            if plateau_count > 50:
                # Perturb to escape local minimum
                for r in range(1, R):
                    for i in range(n * m):
                        u[r][i] += rng.gauss(0, 0.05)
                    for j in range(m * p):
                        v[r][j] += rng.gauss(0, 0.05)
                    for k in range(p * n):
                        w[r][k] += rng.gauss(0, 0.05)
                plateau_count = 0

    if best_state is None:
        return None

    u, v, w = best_state

    # Round to half-integers and verify
    U_frac = [[_round_half(x) for x in row] for row in u]
    V_frac = [[_round_half(x) for x in row] for row in v]
    W_frac = [[_round_half(x) for x in row] for row in w]

    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
        return {"U": [[float(x) for x in row] for row in U_frac],
                "V": [[float(x) for x in row] for row in V_frac],
                "W": [[float(x) for x in row] for row in W_frac]}

    return None


def construct():
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)

    # Try rank 17 with multiple anchors and seeds
    for anchor_kind in range(4):
        for seed in range(20):
            if deadline_exceeded():
                break
            result = _try_rank_structured(n, m, p, 17, seed, max_iters=300, anchor_kind=anchor_kind)
            if result is not None:
                return result

    # Try rank 16 (more aggressive, fewer terms)
    for anchor_kind in range(4):
        for seed in range(10):
            if deadline_exceeded():
                break
            result = _try_rank_structured(n, m, p, 16, seed, max_iters=200, anchor_kind=anchor_kind)
            if result is not None:
                return result

    return fallback


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

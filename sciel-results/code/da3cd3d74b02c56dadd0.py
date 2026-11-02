"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _rectangle_from_perimeter():
    # perimeter = 4 -> 2*(W+H)=4 -> W+H=2.
    # We'll keep W=H=1 (square) as the base and let radii adapt.
    return 1.0, 1.0


def _max_inscribed_radius(pos, W, H):
    """Max radius each circle can have to stay inside [0,W]x[0,H]."""
    d = np.minimum(np.minimum(pos[:, 0], W - pos[:, 0]),
                   np.minimum(pos[:, 1], H - pos[:, 1]))
    return np.maximum(d, 1e-9)


def _pairwise_dist(pos):
    return cdist(pos, pos)


def _overlap_penalty(pos, r, beta):
    """Smooth (log-sum-exp) penalty for overlaps. Higher = worse."""
    D = _pairwise_dist(pos)
    # margin = distance - ri - rj; overlap if margin < 0
    margin = D - np.outer(r, r)
    np.fill_diagonal(margin, np.inf)
    # soft-min: -logsumexp(beta*margin)/beta approximates min(margin)
    # penalty = sum over pairs of softmax of (-margin)
    m = -margin
    # avoid overflow
    m_max = np.max(m, axis=1, keepdims=True)
    s = np.sum(np.exp(m - m_max), axis=1)
    soft = -np.log(s) + m_max[:, 0]  # this is soft-min of margin per row
    # We want penalty for violations: use soft-max of (-margin) aggregated
    # Simpler: sum of softplus(-margin) approximated
    viol = np.maximum(-margin, 0.0)
    # log-sum-exp of viol for smoothness
    vmax = np.max(viol)
    if vmax < 1e-12:
        return 0.0
    lse = np.log(np.sum(np.exp(viol - vmax))) + vmax
    return lse


def _objective(x, W, H, beta):
    pos = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    # keep inside rectangle (hard-ish via penalty)
    rmax = _max_inscribed_radius(pos, W, H)
    inside_pen = np.sum(np.maximum(r - rmax, 0.0) ** 2)
    overlap = _overlap_penalty(pos, r, beta)
    # maximize sum r -> minimize -sum r + penalties
    return -np.sum(r) + beta * overlap + 1e3 * inside_pen


def _delaunay_neighbors(pos):
    """Return list of (i,j) pairs that are Delaunay neighbors (reduced set)."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(len(simplex)):
                for b in range(a + 1, len(simplex)):
                    i, j = simplex[a], simplex[b]
                    if i != j:
                        edges.add((min(i, j), max(i, j)))
        return list(edges)
    except Exception:
        # fallback to all pairs
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _overlap_penalty_reduced(pos, r, beta, edges):
    if not edges:
        return 0.0
    viol = np.array([max(pos[i] - pos[j]) for i, j in edges])  # placeholder
    # compute actual margins
    margins = []
    for i, j in edges:
        d = np.linalg.norm(pos[i] - pos[j])
        margins.append(d - r[i] - r[j])
    margins = np.array(margins)
    viol = np.maximum(-margins, 0.0)
    if len(viol) == 0 or np.max(viol) < 1e-12:
        return 0.0
    vmax = np.max(viol)
    lse = np.log(np.sum(np.exp(viol - vmax))) + vmax
    return lse


def _objective_reduced(x, W, H, beta, edges):
    pos = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    rmax = _max_inscribed_radius(pos, W, H)
    inside_pen = np.sum(np.maximum(r - rmax, 0.0) ** 2)
    overlap = _overlap_penalty_reduced(pos, r, beta, edges)
    return -np.sum(r) + beta * overlap + 1e3 * inside_pen


def _grid_init(W, H):
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side * W
            cy = (j + 0.5) / side * H
            cells.append([cx, cy])
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    rmax = _max_inscribed_radius(pos, W, H)
    r = 0.85 * rmax
    return pos, r


def _build_x(pos, r):
    return np.concatenate([pos.ravel(), r])


def construct():
    W, H = _rectangle_from_perimeter()
    best_x = None
    best_val = -np.inf

    pos0, r0 = _grid_init(W, H)
    x0 = _build_x(pos0, r0)

    # Phase 1: coarse smooth optimization with moderate beta
    for beta in [50.0, 200.0, 800.0]:
        if _time_left() < 20.0:
            break
        try:
            res = minimize(_objective, x0, args=(W, H, beta),
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            x0 = res.x
            val = np.sum(x0[2 * N:])
            if val > best_val:
                best_val = val
                best_x = x0.copy()
        except Exception:
            pass

    # Phase 2: ultra-polish with Delaunay-reduced constraint set
    if _time_left() > 15.0:
        try:
            pos = best_x[:2 * N].reshape(N, 2)
            edges = _delaunay_neighbors(pos)
            for beta in [800.0, 2000.0]:
                if _time_left() < 10.0:
                    break
                res = minimize(_objective_reduced, best_x, args=(W, H, beta, edges),
                               method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                best_x = res.x
                val = np.sum(best_x[2 * N:])
                if val > best_val:
                    best_val = val
        except Exception:
            pass

    # Final validity clamp: ensure radii within inscribed and circles don't overlap
    pos = best_x[:2 * N].reshape(N, 2)
    r = best_x[2 * N:].copy()
    rmax = _max_inscribed_radius(pos, W, H)
    r = np.minimum(r, rmax)
    # iterative overlap resolution
    for _ in range(200):
        if _time_left() < 5.0:
            break
        changed = False
        D = _pairwise_dist(pos)
        for i in range(N):
            for j in range(i + 1, N):
                d = D[i, j]
                overlap = r[i] + r[j] - d
                if overlap > 1e-9:
                    # reduce both radii proportionally
                    total = r[i] + r[j]
                    if total < 1e-12:
                        continue
                    frac = overlap / total
                    r[i] *= (1 - frac)
                    r[j] *= (1 - frac)
                    changed = True
        if not changed:
            break

    out = np.zeros((N, 3))
    out[:, :2] = pos
    out[:, 2] = r
    return out
# EVOLVE-BLOCK-END

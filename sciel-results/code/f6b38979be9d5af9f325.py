"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _hard_radius(pts):
    """Compute the true packing radius (min of half pairwise dists and boundary dists)."""
    n = len(pts)
    if n < 2:
        b = 0.5 - np.abs(pts).max(axis=0)
        return float(b.min())
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=0.5, output_type='ndarray')
    if len(pairs) == 0:
        b = 0.5 - np.abs(pts).max(axis=0)
        return float(b.min())
    # Pairwise distances
    diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
    dists = np.linalg.norm(diffs, axis=1)
    min_pair = dists.min()
    # Boundary distances
    bdist = np.minimum(0.5 - np.abs(pts), axis=1)
    min_bound = bdist.min()
    return float(min(min_pair / 2.0, min_bound))


def _grid_init():
    """Standard grid initialisation in the unit square centred at origin."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _soft_min_obj(pts, beta):
    """
    Objective to minimise: negative soft-min of all pairwise half-distances
    and boundary distances.
    """
    n = pts.shape[0]
    # Flatten to (n,2)
    P = pts.reshape(-1, 2)

    # Boundary distances: for each point, min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    bd = np.minimum(0.5 - P, P + 0.5)  # shape (n,2)
    bmin = bd.min(axis=1)  # shape (n,)

    # Pairwise distances: compute all pairs
    # To save memory for large n, we can do it in blocks, but n=50 is small.
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]  # (n,n,2)
    dist = np.linalg.norm(diff, axis=2)  # (n,n)
    np.fill_diagonal(dist, np.inf)
    # We want half of the minimum pairwise distance
    # Actually the objective is: max min( min_pairwise/2, min_boundary )
    # Let's collect all "contribution" values: for each pair, d/2, and for each point, b_i
    # But that's O(n^2) values. For n=50, that's 1225 pairs + 50 points = 1275 values. Fine.

    # Extract upper triangle distances / 2
    iu = np.triu_indices(n, k=1)
    pair_vals = dist[iu] / 2.0  # shape (n(n-1)/2,)

    all_vals = np.concatenate([pair_vals, bmin])  # shape (n(n-1)/2 + n,)

    # Soft-min: -1/beta * log(sum(exp(-beta * vals)))
    # To avoid overflow/underflow, shift by max
    m = all_vals.max()
    exp_vals = np.exp(-beta * (all_vals - m))
    log_sum = np.log(exp_vals.sum())
    soft_min_val = m - np.log(exp_vals.sum()) / beta  # This is the soft min
    # Wait: soft_min = -1/beta * log(sum(exp(-beta * x)))
    # = -1/beta * log( exp(-beta*m) * sum(exp(-beta*(x-m))) )
    # = -1/beta * ( -beta*m + log(sum(exp(-beta*(x-m)))) )
    # = m - 1/beta * log(sum(exp(-beta*(x-m))))
    # We want to MAXIMIZE soft_min, so MINIMIZE -soft_min.
    return -soft_min_val


def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s

    # Start with the grid
    best_pts = _grid_init()
    best_r = _hard_radius(best_pts)

    # Precompute some constants
    beta_start = 10.0
    beta_end = 50.0

    # Try several random restarts with jittered grid
    rng = np.random.default_rng(42)

    for restart in range(20):
        if time.time() > deadline:
            break

        # Jittered grid
        pts = _grid_init()
        jitter = 0.02
        pts += rng.uniform(-jitter, jitter, size=pts.shape)
        # Clamp to stay within [-0.45, 0.45] to keep room for movement
        pts = np.clip(pts, -0.45, 0.45)

        # Run L-BFGS-B with bounds [-0.49, 0.49]
        bounds = [(-0.49, 0.49)] * (2 * N)

        def obj(x):
            return _soft_min_obj(x, beta_start)

        try:
            res = minimize(obj, pts.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        except Exception:
            continue

        if time.time() > deadline:
            break

        cand = res.x.reshape(-1, 2)
        r = _hard_radius(cand)
        if r > best_r:
            best_r = r
            best_pts = cand.copy()

    # Final polish: use a higher beta for a more precise local optimisation on the best found
    if time.time() < deadline - 5.0:
        # Run a few polish iterations with increasing beta
        current = best_pts.copy()
        for beta in [30.0, 60.0, 100.0]:
            if time.time() > deadline:
                break

            def obj_polish(x):
                return _soft_min_obj(x, beta)

            bounds = [(-0.499, 0.499)] * (2 * N)
            try:
                res = minimize(obj_polish, current.flatten(), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 1000,
                                                       'ftol': 1e-14, 'gtol': 1e-12})
            except Exception:
                break
            if time.time() > deadline:
                break
            cand = res.x.reshape(-1, 2)
            r = _hard_radius(cand)
            if r > _hard_radius(current):
                current = cand
                best_pts = cand.copy()
                best_r = r

    # Final check: ensure all points are strictly inside [-0.5, 0.5]
    # Clip to a tiny margin inside the boundary to avoid floating-point issues
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    return best_pts
# EVOLVE-BLOCK-END

"""Pack 50 equal circles in the unit disk, maximizing the common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0  # safety margin before the 150s kill


def _score(pts):
    """Compute the achieved radius for a set of centres."""
    # Distance from each centre to the boundary
    dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    r_boundary = np.min(dists_boundary)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    return min(r_boundary, r_pair)


def _feasible_radius(pts, r):
    """Return violation magnitude: 0 if feasible at radius r, else positive."""
    # Boundary violations
    b = np.linalg.norm(pts, axis=1) + r - 1.0
    # Pairwise violations
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    p = d - 2.0 * r
    # Soft-max to get a smooth violation measure
    # Use a log-sum-exp style aggregation
    b_pos = np.maximum(b, 0.0)
    p_pos = np.maximum(p, 0.0)
    # Sum of violations (smooth enough for L-BFGS-B)
    return np.sum(b_pos) + np.sum(p_pos) / 2.0


def _optimize_for_radius(pts, r):
    """Try to make pts feasible at radius r via L-BFGS-B on violation."""
    def objective(x):
        p = x.reshape(-1, 2)
        return _feasible_radius(p, r)

    def jac(x):
        # Numerical gradient (Jacobian) for simplicity; could be analytical
        p = x.reshape(-1, 2)
        eps = 1e-7
        f0 = _feasible_radius(p, r)
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            g[i] = (_feasible_radius(xp.reshape(-1, 2), r) - f0) / eps
        return g

    x0 = pts.flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    except Exception:
        return pts, False
    return res.x.reshape(-1, 2), res.fun <= 1e-8


def _initial_hexagonal():
    """Generate a hexagonal lattice initial configuration."""
    # Start with a reasonable spacing
    best_pts = None
    for spacing in np.linspace(0.25, 0.35, 20):
        pts = []
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) >= N:
            # Sort by distance from center, take closest N
            r = np.linalg.norm(pts, axis=1)
            idx = np.argsort(r, kind='stable')[:N]
            if best_pts is None or _score(pts[idx]) > _score(best_pts):
                best_pts = pts[idx]
    if best_pts is None:
        # Fallback: random
        best_pts = np.random.randn(N, 2) * 0.8
    return best_pts


def _random_init():
    """Random initial configuration inside the unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _local_refine(pts, r):
    """Local optimization to improve the radius from a starting point."""
    current_r = _score(pts)
    x = pts.flatten()

    # Define the objective: maximize min distance (negative for minimization)
    def neg_min_dist(x):
        p = x.reshape(-1, 2)
        return -_score(p)

    # Use L-BFGS-B to minimize negative score
    try:
        res = minimize(neg_min_dist, x, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.x is not None:
            new_pts = res.x.reshape(-1, 2)
            new_r = _score(new_pts)
            if new_r > current_r:
                return new_pts, new_r
    except Exception:
        pass
    return pts, current_r


def construct():
    t0 = time.time()
    best_pts = None
    best_r = 0.0

    # Multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() - t0 > DEADLINE:
            break

        # Alternate between hexagonal and random inits
        if restart % 2 == 0:
            pts = _initial_hexagonal()
        else:
            pts = _random_init()

        # Local refinement
        pts, r = _local_refine(pts, _score(pts))

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

        # Binary search on radius to push higher
        # Try to increase the radius
        low_r = r
        high_r = r * 1.15  # 15% headroom
        for _ in range(15):
            if time.time() - t0 > DEADLINE:
                break
            mid_r = (low_r + high_r) / 2
            feasible_pts, ok = _optimize_for_radius(pts, mid_r)
            if ok:
                # Feasible at mid_r, try higher
                low_r = mid_r
                pts = feasible_pts
                # Update best
                actual_r = _score(feasible_pts)
                if actual_r > best_r:
                    best_r = actual_r
                    best_pts = feasible_pts.copy()
            else:
                # Not feasible, try lower
                high_r = mid_r

    # Final polish on the best configuration
    if best_pts is not None and time.time() - t0 < DEADLINE:
        best_pts, best_r = _local_refine(best_pts, best_r)

    # Ensure validity: no coincident centres, all inside
    if best_pts is None:
        best_pts = _initial_hexagonal()

    # Clamp any points that might be slightly outside
    norms = np.linalg.norm(best_pts, axis=1)
    for i in range(len(best_pts)):
        if norms[i] > 1.0:
            best_pts[i] *= (1.0 / norms[i])

    return best_pts

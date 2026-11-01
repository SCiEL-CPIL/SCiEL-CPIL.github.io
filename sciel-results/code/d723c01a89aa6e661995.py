# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _hex_init(N):
    """Generate hexagonal lattice points, select N closest to origin."""
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    s *= 0.95

    best_points = None
    for attempt in range(20):
        k = int(1.0 / s) + 3
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                continue
            x_offset = (j % 2) * s / 2.0
            for i in range(-k, k + 1):
                x = (i + x_offset) * s
                r = np.hypot(x, y)
                if r <= 1.0 - s / 2.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            if best_points is None:
                best_points = pts[idx]
        s *= 0.98
    return best_points if best_points is not None else np.random.randn(N, 2) * 0.5


def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    n = len(pts)
    if n < 2:
        min_pair = 1.0
    else:
        x = pts[:, 0]
        y = pts[:, 1]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_sq = np.min(dists_sq)
        min_pair = np.sqrt(min_pair_sq) / 2.0

    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists_boundary)

    return min(min_pair, min_boundary)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Hexagonal Init
    pts_hex = _hex_init(N)
    r_hex = _compute_radius(pts_hex)
    best_pts = pts_hex.copy()
    best_r = r_hex

    # 2. Differential Evolution
    # DE is a global optimization method, robust to local minima.
    # It minimizes -r.
    
    def objective(x):
        p = x.reshape(N, 2)
        r = _compute_radius(p)
        return -r

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        # Run DE
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=40, 
            tol=1e-10,
            polish=True,  # Uses L-BFGS at the end
            workers=1
        )
        
        if result.fun < -best_r: # i.e. -r_de < -r_hex => r_de > r_hex
            new_pts = result.x.reshape(N, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
                
    except Exception as e:
        pass

    # 3. Ensure validity
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = dists <= 1.0
    if not np.all(mask):
        for i in range(N):
            if dists[i] > 1.0:
                best_pts[i] = best_pts[i] / dists[i] * 0.999

    return best_pts
# EVOLVE-BLOCK-END

"""Circle packing in unit square: Voronoi-based radii with trust-constr optimization."""
import numpy as np
import time

N = 26
DEADLINE = 140.0  # safety margin before 150s kill

def _voronoi_radii(points):
    """
    Given N points in [0,1]^2, compute for each point the radius of the largest
    circle centered there that does not overlap any other circle and stays in [0,1]^2.
    We use scipy.spatial.Voronoi to get cell edges, then compute min distance from
    each point to its cell boundary (which includes the square boundary).
    """
    from scipy.spatial import Voronoi, Delaunay
    
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.array([])
    if n == 1:
        p = pts[0]
        r = min(p[0], 1.0 - p[0], p[1], 1.0 - p[1])
        return np.array([max(0.0, r)])
    
    # Add boundary points to constrain Voronoi within [0,1]^2
    # We add points far outside so that Voronoi cells are clipped to the square
    # Better approach: use the fact that Voronoi of points in a bounded region
    # gives cells bounded by perpendicular bisectors AND the box boundary.
    # scipy's Voronoi gives unbounded cells. We need to clip to [0,1]^2.
    
    # Alternative: for each point, compute distance to nearest other point / 2,
    # and distance to box boundary. But this is not exactly Voronoi radius.
    # The true Voronoi radius (inscribed circle in cell) is what we want.
    
    # Let's use a more direct approach: for each point, the radius is min over
    # all other points of dist/2, and min distance to box edges.
    # This gives the largest circle that doesn't overlap others and fits in box.
    # This is NOT the same as Voronoi cell inscribed circle in general, but it
    # guarantees non-overlap and containment.
    
    radii = np.empty(n)
    for i in range(n):
        # Distance to box boundary
        r_box = min(pts[i][0], 1.0 - pts[i][0], pts[i][1], 1.0 - pts[i][1])
        
        # Distance to nearest other point
        diffs = pts - pts[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf
        r_other = dists.min() / 2.0
        
        radii[i] = min(r_box, r_other)
    
    return np.maximum(0.0, radii)


def _objective(x):
    """Negative sum of radii (for minimization)."""
    pts = x.reshape(-1, 2)
    radii = _voronoi_radii(pts)
    return -np.sum(radii)


def _grad(x):
    """Numerical gradient of the objective."""
    eps = 1e-5
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (_objective(xp) - _objective(xm)) / (2 * eps)
    return g


def _make_grid_initial():
    """Create a grid-based initial placement."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) < N:
                pts.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pts[:N], dtype=np.float64)


def _make_random_initial(seed=42):
    """Random initial placement with minimum separation."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        for attempt in range(200):
            p = rng.rand(2)
            if i == 0 or np.all(np.linalg.norm(pts[:i] - p, axis=1) > 0.05):
                pts[i] = p
                break
        else:
            pts[i] = rng.rand(2)
    return pts


def _is_valid(result):
    """Check if result is a valid packing."""
    if len(result) != N:
        return False
    if result.shape != (N, 3):
        return False
    if np.any(result[:, 2] < 0):
        return False
    if np.any(result[:, 0] < -1e-9) or np.any(result[:, 0] > 1 + 1e-9):
        return False
    if np.any(result[:, 1] < -1e-9) or np.any(result[:, 1] > 1 + 1e-9):
        return False
    # Check containment
    for i in range(N):
        x, y, r = result[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, r1 = result[i]
            x2, y2, r2 = result[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2 - 1e-9:
                return False
    return True


def _post_process(pts):
    """Convert points to (x, y, r) array with computed radii."""
    radii = _voronoi_radii(pts)
    result = np.column_stack([pts, radii])
    return result


def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = -np.inf
    
    def try_optimize(pts0, max_iters=2000):
        nonlocal best_pts, best_score, start_time
        from scipy.optimize import minimize
        
        x0 = pts0.flatten()
        
        def objective(x):
            return _objective(x)
        
        def jac(x):
            return _grad(x)
        
        # Bounds: points in [0,1]^2
        bounds = [(0.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                objective, x0, jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'maxfun': 5000, 'gtol': 1e-8}
            )
        except Exception:
            pass
            return
        
        pts = res.x.reshape(-1, 2)
        radii = _voronoi_radii(pts)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Try multiple initializations
    inits = []
    inits.append(_make_grid_initial())
    for seed in [0, 1, 2, 3, 4]:
        inits.append(_make_random_initial(seed))
    
    for idx, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        try_optimize(init)
    
    # Local refinement: random perturbations around best
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        rng = np.random.RandomState(123)
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            perturbed = best_pts + rng.randn(N, 2) * 0.01
            perturbed = np.clip(perturbed, 0, 1)
            try_optimize(perturbed, max_iters=500)
    
    if best_pts is None:
        best_pts = _make_grid_initial()
    
    result = _post_process(best_pts)
    
    # Final validation
    if not _is_valid(result):
        # Fallback to grid
        result = _post_process(_make_grid_initial())
    
    return result


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
